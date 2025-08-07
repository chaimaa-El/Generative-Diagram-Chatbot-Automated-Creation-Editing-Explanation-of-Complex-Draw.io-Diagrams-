import requests
import urllib3

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY = "sk-EK-xtT3WQ6J3zrWG6ihjxA"  # Replace with your actual API key
BASE_URL = "https://llms-inference.innkube.fim.uni-passau.de"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "llama3.1",  # Use the appropriate model name as per documentation
    "messages": [
        {
            "role": "user",
            "content": "Create a diagram for a simple online shopping flow: Browse -> Add to cart -> Checkout -> Payment."
        }
    ]
}

response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, verify=False)

print("Status code:", response.status_code)
print("Response:", response.json())

