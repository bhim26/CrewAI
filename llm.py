import os

import requests
from dotenv import load_dotenv

load_dotenv()


API_TOKEN = os.getenv("HF_token")
model_id = "meta-llama/Llama-2-7b-chat-hf"
api_url = f"https://api-inference.huggingface.co/models/{model_id}"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


# Define llm as a function
def llm(prompt, temperature=0.7, max_new_tokens=200):
    data = {
        "inputs": prompt,
        "parameters": {"temperature": temperature, "max_new_tokens": max_new_tokens},
    }
    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        raise Exception(f"API call failed: {response.status_code}, {response.text}")
