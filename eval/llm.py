import time
import requests

class LlamaLLM:
    def __init__(self, model_path: str):
        self.model_name = model_path
        self.port = 8000 
        try:
            if requests.get("http://localhost:11434/api/tags", timeout=1).status_code == 200:
                self.port = 11434
        except: pass
        
        self.api_url = f"http://localhost:{self.port}/v1/completions"
        print(f"Connected to port {self.port} with model {self.model_name}")

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.0) -> tuple[str, int, float]:
        t0 = time.time()
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature
        }
        try:
            res = requests.post(self.api_url, json=payload, timeout=300)
            data = res.json()
            response_text = data["choices"][0]["text"].strip()
            tokens = data.get("usage", {}).get("completion_tokens", 0)
        except Exception as e:
            print(f"Error: {e}"); response_text, tokens = "", 0
            
        latency = (time.time() - t0) * 1000.0
        return response_text, tokens, latency