# import json
# import os
# import re
# import codecs
# import xml.etree.ElementTree as ET
# from typing import Dict, List, Optional, Literal
# import requests
# import urllib3
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# class DrawIOGenerator:
#     """LLM-powered Draw.io diagram generator/editor."""
    
#     def __init__(self):
#         self.base_url = "https://llms-inference.innkube.fim.uni-passau.de"
#         self.headers = {
#             "Authorization": f"Bearer {os.getenv('LLM_API_KEY')}",
#             "Content-Type": "application/json"
#         }
    
#     def call_llm(self, prompt: str, model: str = "llama3.1") -> str:
#         """Call the LLM API with the given prompt."""
#         payload = {
#             "model": model,
#             "messages": [{"role": "user", "content": prompt}]
#         }
        
#         try:
#             response = requests.post(
#                 f"{self.base_url}/chat/completions",
#                 headers=self.headers,
#                 json=payload,
#                 verify=False
#             )
#             response.raise_for_status()
#             return response.json()["choices"][0]["message"]["content"]
        
#         except requests.exceptions.RequestException as e:
#             raise RuntimeError(f"API call failed: {str(e)}")

#     @staticmethod
#     def clean_model_output(raw_output: str) -> str:
#         """Extract and clean Draw.io XML from model output."""
#         try:
#             # Remove markdown code blocks if present
#             cleaned = re.sub(r'```(xml)?', '', raw_output).strip()
            
#             # Decode escaped characters
#             decoded = codecs.decode(cleaned, 'unicode_escape')
            
#             # Extract the first valid XML block
#             match = re.search(r"<(mxGraphModel|mxfile)[\s\S]+?</\1>", decoded)
#             return match.group(0) if match else decoded
            
#         except Exception as e:
#             raise ValueError(f"Output cleaning failed: {str(e)}")

#     @staticmethod
#     def validate_drawio_xml(xml_str: str) -> bool:
#         """Validate basic Draw.io XML structure."""
#         try:
#             root = ET.fromstring(xml_str)
#             return root.tag in ("mxGraphModel", "mxfile")
#         except ET.ParseError:
#             return False

#     def process_sample(self, sample: Dict) -> Dict:
#         """Process a single instruction sample."""
#         intent = sample["intent"]
#         instruction = sample["instruction"]
        
#         try:
#             if intent == "generate":
#                 prompt = f"Generate Draw.io XML for: {instruction}"
#             elif intent == "edit":
#                 prompt = f"Edit this Draw.io XML:\n{sample['input_xml']}\n\nInstruction: {instruction}"
#             else:
#                 raise ValueError(f"Unknown intent: {intent}")
            
#             raw_output = self.call_llm(prompt)
#             cleaned_xml = self.clean_model_output(raw_output)
            
#             if not self.validate_drawio_xml(cleaned_xml):
#                 raise ValueError("Invalid Draw.io XML generated")
                
#             return {
#                 "status": "success",
#                 "output": cleaned_xml,
#                 "input_sample": sample
#             }
            
#         except Exception as e:
#             return {
#                 "status": "error",
#                 "error": str(e),
#                 "input_sample": sample
#             }

#     def run_batch(self, input_path: str, output_path: str):
#         """Process all samples from a JSON file."""
#         with open(input_path, "r", encoding="utf-8") as f:
#             samples = json.load(f)
        
#         results = [self.process_sample(sample) for sample in samples]
        
#         with open(output_path, "w", encoding="utf-8") as f:
#             json.dump(results, f, indent=4, ensure_ascii=False)


# if __name__ == "__main__":
#     generator = DrawIOGenerator()
#     generator.run_batch(
#         input_path="../data/sample.json",
#         output_path="../data/llama3_test_results.json"
#     )









import json
import requests
import urllib3
import codecs
import re
import xml.etree.ElementTree as ET
import base64
import zlib



# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY = "sk-EK-xtT3WQ6J3zrWG6ihjxA"
BASE_URL = "https://llms-inference.innkube.fim.uni-passau.de"
MODEL_NAME = "llama3.1"
# MODEL_NAME = "deepseekr1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Function to clean model output and extract XML
def clean_model_output(raw_output):
    try:
        # Remove any backticks and whitespace
        raw_output = raw_output.strip().strip("`")

        # Decode escaped characters
        decoded = codecs.decode(raw_output, 'unicode_escape')

        # Extract <mxGraphModel> or <mxfile> block if present
        match = re.search(r"<(mxGraphModel|mxfile)[\s\S]+?</\1>", decoded)
        return match.group(0) if match else decoded
    except Exception as e:
        print(f"[clean_model_output] Error: {e}")
        return raw_output  # Return original if anything fails
    
def make_drawio_link(xml_str):
    try:
        # Clean and compress
        compressed = zlib.compress(xml_str.encode("utf-8"), level=9)
        # Remove zlib headers (draw.io expects raw DEFLATE, not zlib format)
        deflated = compressed[2:-4]
        encoded = base64.b64encode(deflated).decode("utf-8")
        url = f"https://app.diagrams.net/?lightbox=1&edit=_blank&layers=1&nav=1#R{encoded}"
        return url
    except Exception as e:
        print(f"[make_drawio_link] Error: {e}")
        return None


# Load prompts
with open("../data/generate.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

results = []

for sample in samples:
    intent = sample["intent"]
    instruction = sample["instruction"]

    if intent == "generate":
        user_prompt = f"Instruction: {instruction}\nTask: Generate the draw.io XML that represents this diagram."
    elif intent == "edit":
        input_xml = sample["input_xml"]
        user_prompt = f"Original XML:\n{input_xml}\n\nInstruction: {instruction}\nTask: Return the modified XML after applying the instruction."
    else:
        continue

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, verify=False)

        
        print(f"Raw API response: {response.text}")

        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")

        response_json = response.json()
        raw_output = response_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if not raw_output:
            raise Exception("Empty model output")

        content = clean_model_output(raw_output)

    except Exception as e:
        content = f"Error: {str(e)}"
        print(f"Failed for instruction: {instruction}\nError: {e}")


        # print("\n==============================")
        # print("Instruction:", instruction)
        # print("Model Output:\n" + content)
        # expected = sample.get("xml") if intent == "generate" else sample.get("output_xml")
        # print("Expected Output:\n" + expected)
        # print("==============================\n")

    except Exception as e:
        content = f"Error: {str(e)}"
        print(f"Failed for instruction: {instruction}\nError: {e}")


    expected = sample.get("xml") if intent == "generate" else sample.get("output_xml")

    def normalize_xml(xml_str):
        try:
            root = ET.fromstring(xml_str)
            return ET.tostring(root, encoding="unicode", method="xml").strip()
        except Exception:
            return None

    def classify_output(generated, expected):
        if not generated or not expected:
            return "invalid_xml"
        
        # Normalize whitespace
        if re.sub(r"\s+", "", generated) == re.sub(r"\s+", "", expected):
            return "match"

        # Try XML-aware comparison
        norm_gen = normalize_xml(generated)
        norm_exp = normalize_xml(expected)
        
        if not norm_gen or not norm_exp:
            return "invalid_xml"
        
        if norm_gen == norm_exp:
            return "minor_diff"
        else:
            return "structural_diff"

    tag = classify_output(content, expected)

    drawio_link = make_drawio_link(content) if not content.startswith("Error") else None


    results.append({
        "intent": intent,
        "instruction": instruction,
        "model_output": content,
        "expected_output": expected,
        "tag": tag,
        "drawio_link": drawio_link
    })



# Save results to file
with open("../data/llama3_test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
