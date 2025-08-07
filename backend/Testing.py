import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import zlib
import base64
import urllib.parse
import re



def generate_drawio_link(xml_content):
    if "<mxfile>" in xml_content:
        wrapped = xml_content
    else:
        wrapped = f"""<mxfile><diagram name="Page-1">{xml_content}</diagram></mxfile>"""

    compressed = zlib.compress(wrapped.encode("utf-8"), level=9)[2:-4]
    encoded = base64.b64encode(compressed).decode("utf-8")
    encoded_url = urllib.parse.quote(encoded, safe="")
    return f"https://app.diagrams.net/?lightbox=1&edit=_blank#R{encoded_url}"


def extract_xml(text: str) -> str:
   
    match = re.search(r"(<mxGraphModel>.*?</mxGraphModel>)", text, re.DOTALL)
    if match:
        return f"<mxfile><diagram name=\"AutoDiagram\">{match.group(1).strip()}</diagram></mxfile>"

    
    match = re.search(r"<mxgraph>(.*?)</mxgraph>", text, re.DOTALL)
    if match:
        inner = match.group(1).strip()
        
        return f"<mxfile><diagram name=\"AutoDiagram\">{inner}</diagram></mxfile>"

    raise ValueError("No recognizable diagram XML found in response.")


import re

def sanitize_xml(xml: str) -> str:
    xml = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", 
        "", 
        xml
    )
    return xml.strip()


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="qwen2.5-lora-diagram-chatbot-final",
    max_seq_length=2048,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)  


tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")


test_cases = [
    {
        "intent": "generate",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Generate an XML code for a hierarchy chart:  Principal → Senior Instructor → Associate Lecturer → Learner"
                )
            }
        ]
    },
    {
  "intent": "generate",
  "messages": [
    {
      "role": "user",
      "content": "Generate XML drawio diagram for a flowchart for the online shopping process from product selection to order delivery."
    }
  ]
},

   {
    "intent": "edit",
    "messages": [
        {
            "role": "user",
            "content": (
                "Move the Generator and Discriminator 100 pixels to the right and increase the size of the Discriminator by 50%.\n\n"
                "Input XML:\n"
                "<mxfile host=\"app.diagrams.net\">\n  <diagram name=\"GAN Diagram\" id=\"GAN1\">\n    <mxGraphModel dx=\"1100\" dy=\"800\" grid=\"1\" gridSize=\"10\" guides=\"1\" tooltips=\"1\" connect=\"1\" arrows=\"1\" fold=\"1\" page=\"1\" pageScale=\"1\" pageWidth=\"850\" pageHeight=\"1100\">\n      <root>\n        <mxCell id=\"0\"/>\n        <mxCell id=\"1\" parent=\"0\"/>\n        <mxCell id=\"2\" value=\"Noise (z)\" style=\"ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"20\" y=\"140\" width=\"100\" height=\"60\" as=\"geometry\"/>\n        </mxCell>\n        <mxCell id=\"3\" value=\"Generator\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"150\" y=\"120\" width=\"120\" height=\"100\" as=\"geometry\"/>\n        </mxCell>\n        <mxCell id=\"4\" value=\"Fake Data (x~)\" style=\"rounded=0;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"300\" y=\"140\" width=\"100\" height=\"60\" as=\"geometry\"/>\n        </mxCell>\n        <mxCell id=\"5\" value=\"Discriminator\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"440\" y=\"100\" width=\"140\" height=\"120\" as=\"geometry\"/>\n        </mxCell>\n        <mxCell id=\"6\" value=\"Real/Fake?\" style=\"rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#000000;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"600\" y=\"140\" width=\"100\" height=\"60\" as=\"geometry\"/>\n        </mxCell>\n        <mxCell id=\"7\" value=\"Real Data (x)\" style=\"rounded=0;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"300\" y=\"40\" width=\"100\" height=\"60\" as=\"geometry\"/>\n        </mxCell>\n        <mxCell id=\"8\" style=\"endArrow=block;html=1;\" edge=\"1\" parent=\"1\" source=\"2\" target=\"3\">\n          <mxGeometry relative=\"1\" as=\"geometry\"/>\n        </mxCell>\n        <mxCell id=\"9\" style=\"endArrow=block;html=1;\" edge=\"1\" parent=\"1\" source=\"3\" target=\"4\">\n          <mxGeometry relative=\"1\" as=\"geometry\"/>\n        </mxCell>\n        <mxCell id=\"10\" style=\"endArrow=block;html=1;\" edge=\"1\" parent=\"1\" source=\"4\" target=\"5\">\n          <mxGeometry relative=\"1\" as=\"geometry\"/>\n        </mxCell>\n        <mxCell id=\"11\" style=\"endArrow=block;html=1;\" edge=\"1\" parent=\"1\" source=\"7\" target=\"5\">\n          <mxGeometry relative=\"1\" as=\"geometry\"/>\n        </mxCell>\n        <mxCell id=\"12\" style=\"endArrow=block;html=1;\" edge=\"1\" parent=\"1\" source=\"5\" target=\"6\">\n          <mxGeometry relative=\"1\" as=\"geometry\"/>\n        </mxCell>\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>"
            )
        }
    ]
}
,
{
        "intent": "edit",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Increase the width of the 'Input Image' block by 50 units and move the 'Task Head B - Regression' blocks 30 units down to create better spacing below the shared layers without overlapping any other elements.\n\n"
                    "Input XML:\n"
                    "<mxfile host=\"app.diagrams.net\">\n  <diagram name=\"Detailed Multi-Task Learning\" id=\"detailedMTL\">\n    <mxGraphModel dx=\"1300\" dy=\"900\" grid=\"1\" gridSize=\"10\" guides=\"1\" tooltips=\"1\" connect=\"1\" arrows=\"1\" fold=\"1\" page=\"1\" pageScale=\"1\" pageWidth=\"850\" pageHeight=\"1100\">\n      <root>\n        <mxCell id=\"0\"/>\n        <mxCell id=\"1\" parent=\"0\"/>\n\n        <!-- Input -->\n        <mxCell id=\"2\" value=\"Input Image\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#000000;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"40\" y=\"180\" width=\"100\" height=\"50\" as=\"geometry\"/>\n        </mxCell>\n\n        <!-- Shared Backbone Layers -->\n        <mxCell id=\"3\" value=\"Conv1\\n(ReLU)\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"160\" y=\"100\" width=\"100\" height=\"50\" as=\"geometry\"/>\n        </mxCell>\n        <mxCell id=\"4\" value=\"Conv2\\n(ReLU + Pool)\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"280\" y=\"100\" width=\"100\" height=\"50\" as=\"geometry\"/>\n        </mxCell>\n        <mxCell id=\"5\" value=\"Conv3\\n(ReLU)\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"400\" y=\"100\" width=\"100\" height=\"50\" as=\"geometry\"/>\n        </mxCell>\n\n        <!-- Shared Backbone Block -->\n        <mxCell id=\"6\" value=\"Shared Feature Extractor\" style=\"swimlane;startSize=20;html=1;fillColor=none;strokeColor=gray;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"150\" y=\"80\" width=\"370\" height=\"100\" as=\"geometry\"/>\n        </mxCell>\n\n        <!-- Task Head A - Classification -->\n        <mxCell id=\"7\" value=\"FC1\\n(ReLU)\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"550\" y=\"40\" width=\"100\" height=\"40\" as=\"geometry\"/>\n        </mxCell>\n        <mxCell id=\"8\" value=\"Softmax\\n(Classification)\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"670\" y=\"40\" width=\"100\" height=\"40\" as=\"geometry\"/>\n        </mxCell>\n\n        <!-- Task Head B - Regression -->\n        <mxCell id=\"9\" value=\"FC1\\n(ReLU)\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"550\" y=\"140\" width=\"100\" height=\"40\" as=\"geometry\"/>\n        </mxCell>\n        <mxCell id=\"10\" value=\"Linear\\n(Regression)\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"670\" y=\"140\" width=\"100\" height=\"40\" as=\"geometry\"/>\n        </mxCell>\n\n        <!-- Task Head C - Segmentation -->\n        <mxCell id=\"11\" value=\"Upsample\\n+ Conv\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"550\" y=\"240\" width=\"100\" height=\"40\" as=\"geometry\"/>\n        </mxCell>\n        <mxCell id=\"12\" value=\"Pixel-wise\\nSoftmax\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"670\" y=\"240\" width=\"100\" height=\"40\" as=\"geometry\"/>\n        </mxCell>\n\n        <!-- Arrows from Input to Backbone -->\n        <mxCell id=\"13\" style=\"endArrow=block;html=1;\" edge=\"1\" parent=\"1\" source=\"2\" target=\"3\">\n          <mxGeometry relative=\"1\" as=\"geometry\"/>\n        </mxCell>\n        <mxCell id=\"14\" style=\"endArrow=block;html=1;\" edge=\"1\" parent=\"1\" source=\"3\" target=\"4\">\n          <mxGeometry relative=\"1\" as=\"geometry\"/>\n        </mxCell>\n        <mxCell id=\"15\" style=\"endArrow=block;html=1;\" edge=\"1\" parent=\"1\" source=\"4\" target=\"5\">\n          <mxGeometry relative=\"1\" as=\"geometry\"/>\n        </mxCell>\n\n        <!-- Arrows from Backbone to Classification Head -->\n        <mxCell id=\"16\" style=\"endArrow=block;html=1;\" edge=\"1\" parent=\"1\" source=\"5\" target=\"7\">\n          <mxGeometry relative=\"1\" as=\"geometry\"/>\n        </mxCell>\n        <mxCell id=\"17\" style=\"endArrow=block;html=1;\" edge=\"1\" parent=\"1\" source=\"7\" target=\"8\">\n          <mxGeometry relative=\"1\" as=\"geometry\"/>\n        </mxCell>\n\n        <!-- Arrows from Backbone to Regression Head -->\n        <mxCell id=\"18\" style=\"endArrow=block;html=1;\" edge=\"1\" parent=\"1\" source=\"5\" target=\"9\">\n          <mxGeometry relative=\"1\" as=\"geometry\"/>\n        </mxCell>\n        <mxCell id=\"19\" style=\"endArrow=block;html=1;\" edge=\"1\" parent=\"1\" source=\"9\" target=\"10\">\n          <mxGeometry relative=\"1\" as=\"geometry\"/>\n        </mxCell>\n\n        <!-- Arrows from Backbone to Segmentation Head -->\n        <mxCell id=\"20\" style=\"endArrow=block;html=1;\" edge=\"1\" parent=\"1\" source=\"5\" target=\"11\">\n          <mxGeometry relative=\"1\" as=\"geometry\"/>\n        </mxCell>\n        <mxCell id=\"21\" style=\"endArrow=block;html=1;\" edge=\"1\" parent=\"1\" source=\"11\" target=\"12\">\n          <mxGeometry relative=\"1\" as=\"geometry\"/>\n        </mxCell>\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n\n"
                )
            }
        ],
          }

,
    {
        "intent": "explain",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Explain the following diagram:\n\n"
                    "<mxfile><diagram name=\"Customer Feedback Loop\">"
                    "<mxGraphModel><root>"
                    "<mxCell id='0'/><mxCell id='1' parent='0'/>"
                    "<mxCell id='2' value='Customer' style='ellipse;fillColor=#C8E6C9;strokeColor=#388E3C' vertex='1' parent='1'>"
                    "<mxGeometry x='100' y='80' width='100' height='50' as='geometry'/></mxCell>"
                    "<mxCell id='3' value='Feedback Portal' style='rectangle;fillColor=#FFECB3;strokeColor=#FF9800' vertex='1' parent='1'>"
                    "<mxGeometry x='250' y='80' width='140' height='50' as='geometry'/></mxCell>"
                    "<mxCell id='4' value='Analytics Engine' style='rectangle;fillColor=#BBDEFB;strokeColor=#2196F3' vertex='1' parent='1'>"
                    "<mxGeometry x='450' y='80' width='160' height='50' as='geometry'/></mxCell>"
                    "<mxCell id='5' value='Product Team' style='ellipse;fillColor=#FFCDD2;strokeColor=#F44336' vertex='1' parent='1'>"
                    "<mxGeometry x='650' y='80' width='120' height='50' as='geometry'/></mxCell>"
                    "<mxCell id='6' edge='1' source='2' target='3' parent='1'><mxGeometry relative='1' as='geometry'/></mxCell>"
                    "<mxCell id='7' edge='1' source='3' target='4' parent='1'><mxGeometry relative='1' as='geometry'/></mxCell>"
                    "<mxCell id='8' edge='1' source='4' target='5' parent='1'><mxGeometry relative='1' as='geometry'/></mxCell>"
                    "</root></mxGraphModel></diagram></mxfile>"
                )
            }
        ]
    }
]




for case in test_cases:
    print(f"\n🚀 Intent: {case['intent']}")
    
    inputs = tokenizer.apply_chat_template(
        case["messages"],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=2048,
        temperature=0.4,
        top_p=0.9,
        use_cache=True,
        do_sample=True,
       
    )

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    assistant_response = response.split("assistant\n")[-1].strip()
    print("Response:\n", assistant_response)
    if case["intent"] in ["generate", "edit"]:
        try:
            if "<mxfile>" in assistant_response:
                raw_xml = assistant_response
            else:
                raw_xml = extract_xml(assistant_response)

            clean_xml = sanitize_xml(raw_xml)
            link = generate_drawio_link(clean_xml)
            print(" Draw.io Link:", link)

        except Exception as e:
            print(f"[!] Failed to generate link: {e}")