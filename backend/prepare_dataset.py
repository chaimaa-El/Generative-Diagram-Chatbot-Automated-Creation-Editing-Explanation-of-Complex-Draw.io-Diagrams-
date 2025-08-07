import json
from datasets import Dataset


def clean_xml(text: str) -> str:
    """Standardize XML by stripping and replacing escaped quotes."""
    return text.strip().replace('\\"', '"')


def load_data(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    formatted = []
    for ex in raw:
        intent = ex.get("intent")

        # === GENERATE ===
        if intent == "generate":
            instruction = ex.get("instruction", "").strip()
            xml_output = clean_xml(ex.get("xml", ""))
            formatted.append({
                "conversations": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": xml_output}
                ],
                "intent": intent
            })

        # === EDIT ===
        elif intent == "edit":
            instruction = ex.get("instruction", "").strip()
            input_xml = clean_xml(ex.get("input_xml", ""))
            output_xml = clean_xml(ex.get("output_xml", ""))
            user_msg = f"{instruction}\n\nInput XML:\n{input_xml}"
            formatted.append({
                "conversations": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": output_xml}
                ],
                "intent": intent
            })

        # === EXPLAIN ===
        elif intent == "explain":
            instruction = ex.get("instruction", "").strip()
            input_xml = clean_xml(ex.get("input_xml", ""))
            explanation = ex.get("explanation", "").strip()
            user_msg = f"{instruction}\n\n{input_xml}"
            formatted.append({
                "conversations": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": explanation}
                ],
                "intent": intent
            })

        # (optional): Add more intents here later

    return Dataset.from_list(formatted)