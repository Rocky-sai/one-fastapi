"""
Simple test to verify Perplexity API integration
"""
import requests
import os
from dotenv import load_dotenv

# Load from backend/.env
load_dotenv('backend/.env')
API_KEY = os.getenv("GEMINI_API_KEY")

def test_perplexity_api():
    if not API_KEY or not API_KEY.startswith("pplx-"):
        print("No Perplexity API key found")
        return
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama-3.1-sonar-small-128k-chat",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful coding assistant."
            },
            {
                "role": "user", 
                "content": "Write a simple Python function to add two numbers"
            }
        ],
        "max_tokens": 200,
        "temperature": 0.2
    }
    
    try:
        response = requests.post("https://api.perplexity.ai/chat/completions", 
                               headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                print("✅ Perplexity API works!")
                print("Response:", result["choices"][0]["message"]["content"][:200])
                return True
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_perplexity_api()
