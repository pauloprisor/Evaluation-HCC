import time
import requests

class LlamaLLM:
    def __init__(self, model_path: str):
        self.model_name = model_path
        self.api_url = "http://localhost:11434/api/generate"
        
        try:
            requests.get("http://localhost:11434/")
            print(f"Ollama connection verified. Bound to model: {self.model_name}")
        except requests.exceptions.ConnectionError:
            print(f"WARNING: Ollama is not running on localhost:11434")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.1
    ) -> tuple[str, int, float]:

        t0 = time.time()
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": max_new_tokens,
            }
        }
        
        try:
            res = requests.post(self.api_url, json=payload, timeout=300)
            res.raise_for_status()
            data = res.json()
            
            response_text = data.get("response", "")
            completion_tokens = data.get("eval_count", 0)  
            
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            response_text = ""
            completion_tokens = 0
            
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000.0
        
        return response_text, completion_tokens, latency_ms
