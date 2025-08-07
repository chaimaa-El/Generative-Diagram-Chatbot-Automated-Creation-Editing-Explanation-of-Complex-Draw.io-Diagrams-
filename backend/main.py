from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import zlib
import base64
import urllib.parse
import re

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import StoppingCriteria, StoppingCriteriaList

# === Load model ===
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="qwen2.5-lora-diagram-chatbot-final",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === App setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# === Custom stop at </mxfile> ===
class StopXML(StoppingCriteria):
    def __init__(self, stop_string, tokenizer):
        self.stop_string = stop_string
        self.tokenizer = tokenizer
        self.buffer = ""

    def __call__(self, input_ids, scores, **kwargs):
        self.buffer += self.tokenizer.decode(input_ids[0, -1], skip_special_tokens=True)
        return self.stop_string in self.buffer

# === Utility functions ===
def generate_drawio_link(xml_content):
    if "<mxfile>" in xml_content:
        wrapped = xml_content
    else:
        wrapped = f"<mxfile><diagram name='Auto'>{xml_content}</diagram></mxfile>"
    compressed = zlib.compress(wrapped.encode("utf-8"), level=9)[2:-4]
    encoded = base64.b64encode(compressed).decode("utf-8")
    encoded_url = urllib.parse.quote(encoded, safe="")
    return f"https://app.diagrams.net/?lightbox=1&edit=_blank#R{encoded_url}"

def extract_last_xml(text: str) -> str:
    # Find all <mxfile>...</mxfile> blocks and return the last one
    matches = re.findall(r"(<mxfile>.*?</mxfile>)", text, re.DOTALL)
    if matches:
        return matches[-1]
    return ""

def extract_explanation(text: str) -> str:
    # Return everything before the first <mxfile>
    parts = re.split(r"<mxfile>", text, maxsplit=1)
    return parts[0].strip() if parts else text.strip()

def sanitize_xml(xml: str) -> str:
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", xml).strip()

def remove_role_echoes(text: str) -> str:
    # Remove role tags like system, user, assistant (case-insensitive, with or without colon)
    return re.sub(r"\b(system|user|assistant)\b\s*:?[\n]*", "", text, flags=re.IGNORECASE).strip()

# === API Schemas ===
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

def detect_explain_intent(user_msg: str) -> bool:
    user_msg = user_msg.lower()
    explain_keywords = ["explain", "describe", "analyze", "what does", "summarize", "interpret", "break down"]
    return any(keyword in user_msg for keyword in explain_keywords)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        user_msg = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        intent = "explain" if detect_explain_intent(user_msg) else None

        prompt = tokenizer.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        if isinstance(inputs, torch.Tensor):
            inputs = {"input_ids": inputs.to(device)}
        else:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        stopping_criteria = StoppingCriteriaList([StopXML("</mxfile>", tokenizer)])

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=2048,
            stopping_criteria=stopping_criteria,
            do_sample=False,
            temperature=0.7,
            top_p=0.95,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print("🚀 Model output:\n", decoded)

        if intent == "explain":
            # Extract everything after the last 'assistant' marker
            match = re.split(r"\bassistant\b[\s:]*", decoded, flags=re.IGNORECASE)
            if len(match) > 1:
                explanation = match[-1].strip()
            else:
                explanation = decoded.strip()
            print("🚀 Explanation extracted:\n", explanation)
            return {
                "role": "assistant",
                "content": explanation,
                "drawio_link": None,
            }

        cleaned = remove_role_echoes(decoded)
        last_xml = extract_last_xml(cleaned)
        response_text = sanitize_xml(last_xml) if last_xml else cleaned
        drawio_url = generate_drawio_link(response_text) if last_xml else None

        return {
            "role": "assistant",
            "content": response_text,
            "drawio_link": drawio_url,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "role": "assistant",
            "content": "Sorry, something went wrong 💔",
            "drawio_link": None
        }