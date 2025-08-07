import base64
import webbrowser
import zlib
import json

def test_drawio_xml(xml_str: str):
    """
    Compresses the XML using raw DEFLATE and opens it in the Draw.io viewer.
    """
    # Compress using raw DEFLATE (wbits=-15 for raw)
    compressed = zlib.compress(xml_str.encode("utf-8"), level=9)[2:-4]  # remove zlib header/footer
    encoded = base64.b64encode(compressed).decode("utf-8")

    url = f"https://viewer.diagrams.net/?highlight=0000ff&edit=_blank&layers=1&nav=1#R{encoded}"
    webbrowser.open(url)
    print("✅ Opened diagram in browser.")

# Example usage
if __name__ == "__main__":
    with open("../data/explain_sample.json", "r", encoding="utf-8") as f:
        sample = json.load(f)
        test_drawio_xml(sample["xml"])




