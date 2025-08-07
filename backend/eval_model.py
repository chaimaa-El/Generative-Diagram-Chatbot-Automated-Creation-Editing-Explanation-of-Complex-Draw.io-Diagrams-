import torch
import networkx as nx
import xml.etree.ElementTree as ET
import re
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_from_disk
import evaluate
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# === Load model & tokenizer ===
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="qwen2.5-lora-diagram-chatbot-final",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === XML Extraction and Sanitization ===
def extract_first_xml(text: str) -> str:
    # Find the first <mxfile>...</mxfile> block
    match = re.search(r"(<mxfile>.*?</mxfile>)", text, re.DOTALL)
    if match:
        return match.group(1)
    # Try <mxGraphModel>
    match = re.search(r"(<mxGraphModel>.*?</mxGraphModel>)", text, re.DOTALL)
    if match:
        return f"<mxfile><diagram name=\"AutoDiagram\">{match.group(1).strip()}</diagram></mxfile>"
    # Try <mxgraph>
    match = re.search(r"<mxgraph>(.*?)</mxgraph>", text, re.DOTALL)
    if match:
        inner = match.group(1).strip()
        return f"<mxfile><diagram name=\"AutoDiagram\">{inner}</diagram></mxfile>"
    raise ValueError("No recognizable diagram XML found in response.")

def sanitize_xml(xml: str) -> str:
    xml = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", xml)
    return xml.strip()

def is_valid_xml(xml: str) -> bool:
    try:
        ET.fromstring(xml)
        return True
    except Exception:
        return False

# === Graph utils ===
def normalize_label(label):
    return label.lower().replace('-', '').replace(':', '').strip()

def parse_xml_to_graph(xml_string):
    G = nx.DiGraph()
    try:
        root = ET.fromstring(xml_string)
    except Exception as e:
        print(f"[!] XML parsing failed: {e}")
        return G
    for cell in root.iter('mxCell'):
        if 'vertex' in cell.attrib:
            node_id = cell.attrib['id']
            label = cell.attrib.get('value', '')
            G.add_node(node_id, label=normalize_label(label))
        elif 'edge' in cell.attrib:
            source = cell.attrib.get('source')
            target = cell.attrib.get('target')
            if source and target:
                G.add_edge(source, target)
    return G

def evaluate_graphs(reference_xml, generated_xml):
    G_ref = parse_xml_to_graph(reference_xml)
    G_gen = parse_xml_to_graph(generated_xml)

    ref_labels = set(nx.get_node_attributes(G_ref, 'label').values())
    gen_labels = set(nx.get_node_attributes(G_gen, 'label').values())
    common_labels = ref_labels & gen_labels

    precision = len(common_labels) / len(gen_labels) if gen_labels else 0
    recall = len(common_labels) / len(ref_labels) if ref_labels else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) else 0

    try:
        ged = nx.graph_edit_distance(G_ref, G_gen)
        max_possible = max(len(G_ref.nodes) + len(G_ref.edges), len(G_gen.nodes) + len(G_gen.edges))
        ged_score = 1 - (ged / max_possible) if max_possible and ged is not None else 1
    except Exception as e:
        print(f"[!] GED failed: {e}")
        ged_score = 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ged_score': ged_score,
    }

# === Metrics ===
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

eval_dataset = load_from_disk("data/eval_dataset")

def evaluate_model(dataset, output_json_path="eval_outputsfin3.json"):
    precision_scores, recall_scores, f1_scores, ged_scores = [], [], [], []
    bleu_scores, rouge_scores = [], []
    per_example_logs = []

    model.eval()
    for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
        t0 = time.time()
        intent = example["intent"]
        reference_text = example["conversations"][-1]["content"].strip()
        if not reference_text:
            continue

        if "text" in example:
            formatted_input = example["text"]
        else:
            formatted_input = tokenizer.apply_chat_template(
                example["conversations"],
                tokenize=False,
                add_generation_prompt=True
            )

        # Tokenize as a list for safety
        inputs = tokenizer(
            [formatted_input],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=512,  # Lowered for speed
                temperature=0.7,
                top_p=0.9,
                use_cache=True,
                do_sample=False  # Deterministic for eval
            )

        pred_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # Print the first 3 outputs for debugging
        if idx < 3:
            print(f"\n[DEBUG] Example {idx} model output:\n{pred_text}\n")

        # Robust assistant response extraction
        if "assistant" in pred_text:
            assistant_response = pred_text.split("assistant", 1)[-1].strip()
        else:
            assistant_response = pred_text.strip()

        # === Extract and validate only the first XML block ===
        try:
            raw_xml = extract_first_xml(assistant_response)
            # Remove any garbage after </mxfile>
            end_idx = raw_xml.find("</mxfile>") + len("</mxfile>")
            clean_xml = sanitize_xml(raw_xml[:end_idx])
            if not is_valid_xml(clean_xml):
                raise ValueError("Generated XML is not valid.")
        except Exception as e:
            print(f"[!] XML extraction/validation failed: {e}")
            precision_scores.append(0)
            recall_scores.append(0)
            f1_scores.append(0)
            ged_scores.append(0)
            per_example_logs.append({
                "intent": intent,
                "input": formatted_input,
                "prediction": assistant_response,
                "reference": reference_text,
                "metrics": {
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                    "ged_score": 0,
                    "error": str(e),
                }
            })
            continue

        log = {
            "intent": intent,
            "input": formatted_input,
            "prediction": assistant_response,
            "reference": reference_text,
            "metrics": {}
        }

        if intent in ["generate", "edit"]:
            try:
                scores = evaluate_graphs(reference_text, clean_xml)
                precision_scores.append(scores["precision"])
                recall_scores.append(scores["recall"])
                f1_scores.append(scores["f1"])
                ged_scores.append(scores["ged_score"])
                log["metrics"] = scores
            except Exception as e:
                print(f"[!] Graph eval failed: {e}")
                precision_scores.append(0)
                recall_scores.append(0)
                f1_scores.append(0)
                ged_scores.append(0)
                log["metrics"] = {
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                    "ged_score": 0,
                    "error": str(e),
                }

        elif intent == "explain":
            try:
                bleu = bleu_metric.compute(predictions=[assistant_response], references=[reference_text])
                rouge = rouge_metric.compute(predictions=[assistant_response], references=[reference_text])

                bleu_scores.append(bleu["bleu"])
                rouge_scores.append(rouge["rouge1"])

                log["metrics"] = {
                    "bleu": bleu["bleu"],
                    "rouge1": rouge["rouge1"]
                }
            except Exception as e:
                print(f"[!] Text eval failed: {e}")
                log["metrics"] = {"bleu": 0, "rouge1": 0, "error": str(e)}

        per_example_logs.append(log)
        t1 = time.time()
        if idx < 3:
            print(f"[DEBUG] Example {idx} took {t1-t0:.2f} seconds")

    # === Write detailed results to JSON ===
    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(per_example_logs, f, indent=2, ensure_ascii=False)

    # === Compute and return summary ===
    results = {}
    if precision_scores:
        results["graph_precision"] = sum(precision_scores) / len(precision_scores)
        results["graph_recall"] = sum(recall_scores) / len(recall_scores)
        results["graph_f1"] = sum(f1_scores) / len(f1_scores)
    if ged_scores:
        results["graph_ged_score"] = sum(ged_scores) / len(ged_scores)
    if bleu_scores:
        results["bleu"] = sum(bleu_scores) / len(bleu_scores)
    if rouge_scores:
        results["rouge1_f1"] = sum(rouge_scores) / len(rouge_scores)

    return results, f1_scores, ged_scores, bleu_scores, rouge_scores

results, f1_scores, ged_scores, bleu_scores, rouge_scores = evaluate_model(eval_dataset)
print("Evaluation Results:", results)

def plot_metric(values, title, ylabel, filename):
    if not values: return
    plt.figure(figsize=(8, 5))
    plt.plot(values, marker='o')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Example Index")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_metric(f1_scores, "Graph F1 Scores", "F1", "f1_scores.png")
plot_metric(ged_scores, "Graph Edit Distance Scores", "GED", "ged_scores.png")
plot_metric(bleu_scores, "BLEU Scores", "BLEU", "bleu_scores.png")
plot_metric(rouge_scores, "ROUGE-1 F1 Scores", "ROUGE-1", "rouge_scores.png")
print("Saved plots for F1, GED, BLEU, and ROUGE")