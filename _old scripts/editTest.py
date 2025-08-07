import base64
import webbrowser
import zlib
import json

def test_drawio_xml(xml_str: str, title: str = "diagram"):
    """
    Compresses the XML using raw DEFLATE and opens it in the Draw.io viewer.
    """
    compressed = zlib.compress(xml_str.encode("utf-8"), level=9)[2:-4]  # raw DEFLATE (no zlib header/footer)
    encoded = base64.b64encode(compressed).decode("utf-8")

    url = f"https://viewer.diagrams.net/?highlight=0000ff&edit=_blank&layers=1&nav=1#R{encoded}"
    webbrowser.open_new_tab(url)
    print(f"✅ Opened {title} diagram in browser.")

# Example usage
if __name__ == "__main__":
    with open("../data/explain_sample.json", "r", encoding="utf-8") as f:
        sample = json.load(f)

        if "input_xml" in sample:
            test_drawio_xml(sample["input_xml"], title="input_xml")

        if "output_xml" in sample:
            test_drawio_xml(sample["output_xml"], title="output_xml")