import time
import torch
import warnings
from typing import Tuple

from .base import BaseMethod, count_tokens

try:
    from llmlingua import PromptCompressor
except ImportError:
    PromptCompressor = None

class LLMLingua1Compressor(BaseMethod):
    def __init__(self, **kwargs):
        if PromptCompressor is None:
            raise ImportError("Please install llmlingua package first.")
        
        # Forțăm CPU pentru LLMLingua v1 (evită erorile de 'Invalid buffer size' pe MPS la contexte lungi)
        self.device = "cpu"
        hf_id = "Qwen/Qwen2.5-0.5B"
        
        print(f"[LLMLingua-1] Setup: Loading SLM {hf_id} on {self.device}...")
        self.compressor = PromptCompressor(
            model_name=hf_id,
            use_llmlingua2=False,
            device_map=self.device,
            model_config={
                "trust_remote_code": True
            }
        )

    def get_name(self) -> str:
        return "llmlingua"

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
            
        target_ratio = min(1.0, float(token_budget) / float(original_tokens))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.compressor.compress_prompt(
                context=context,
                instruction="", 
                question=question, 
                target_token=token_budget,
                rate=target_ratio,
                force_tokens=["\n", ".", "!", "?", ","], 
            )
        
        compressed_text = result["compressed_prompt"]
        compressed_tokens = count_tokens(compressed_text)
        
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000.0
        
        return compressed_text, original_tokens, compressed_tokens, latency_ms
