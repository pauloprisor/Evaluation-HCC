import sys
import os
import time
import requests
import subprocess
from typing import Tuple

from .base import BaseMethod, count_tokens

class CPCCompressorMethod(BaseMethod):
    def __init__(self, **kwargs):
        print("[CPC] Setup: Starting isolated CPC server...")
        
        cpc_repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'cpc_repo'))
        venv_python = os.path.abspath(os.path.join(cpc_repo_path, '..', 'cpc_venv', 'bin', 'python'))
        server_script = os.path.join(os.path.dirname(__file__), 'cpc_server.py')
        
        if not os.path.exists(venv_python):
            raise RuntimeError(f"CPC venv not found at {venv_python}. Please create it first.")
            
        log_file_path = os.path.join(cpc_repo_path, '..', 'cpc_server_run.log')
        self.server_log_file = open(log_file_path, 'w')
        
        self.server_process = subprocess.Popen(
            [venv_python, server_script],
            stdout=self.server_log_file,
            stderr=subprocess.STDOUT
        )
        
        self.server_url = "http://127.0.0.1:5001"
        
        # Wait for server to boot and model to load
        max_retries = 120
        ready = False
        for i in range(max_retries):
            try:
                res = requests.get(f"{self.server_url}/ping", timeout=2)
                if res.status_code == 200:
                    ready = True
                    break
            except requests.exceptions.RequestException:
                time.sleep(1)
                
        if not ready:
            self.server_process.kill()
            raise RuntimeError("CPC server failed to start within the timeout period.")
            
        print("[CPC] Server is ready!")

    def get_name(self) -> str:
        return "cpc"

    def compress(
        self,
        context: str,
        question: str,
        token_budget: int
    ) -> Tuple[str, int, int, float]:
        
        t0 = time.time()
        original_tokens = count_tokens(context)
        
        if original_tokens <= token_budget:
            t1 = time.time()
            return context, original_tokens, original_tokens, (t1 - t0) * 1000.0
            
        # Send HTTP request to local server
        payload = {
            "context": context,
            "question": question,
            "token_budget": token_budget
        }
        
        try:
            res = requests.post(f"{self.server_url}/compress", json=payload)
            res.raise_for_status()
            data = res.json()
            if "error" in data:
                raise RuntimeError(data["error"])
            compressed_text = data["compressed_text"]
        except Exception as e:
            raise RuntimeError(f"Failed to compress via CPC server: {e}")
            
        compressed_tokens = count_tokens(compressed_text)
        
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000.0
        
        return compressed_text, original_tokens, compressed_tokens, latency_ms
        
    def __del__(self):
        if hasattr(self, 'server_process') and self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
        if hasattr(self, 'server_log_file') and self.server_log_file:
            self.server_log_file.close()

