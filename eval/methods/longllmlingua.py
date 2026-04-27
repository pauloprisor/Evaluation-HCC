import time
import torch
import warnings
from typing import Tuple

from .base import BaseMethod, count_tokens

try:
    from llmlingua import PromptCompressor
except ImportError:
    PromptCompressor = None

class LongLLMLinguaCompressor(BaseMethod):
    def __init__(self, **kwargs):
        if PromptCompressor is None:
            raise ImportError(" llmlingua package not installed.")
        
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        hf_model_id = "gpt2"
        
        print(f"[LongLLMLingua] Setup: Loading SLM {hf_model_id} pe {self.device}...")
        self.compressor = PromptCompressor(
            model_name=hf_model_id,
            use_llmlingua2=False,
            device_map=self.device
        )

    def get_name(self) -> str:
        return "longllmlingua"

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
                [context],
                instruction="",
                question=question,
                target_token=token_budget,
                rank_method="longllmlingua",
                iterative_size=200,                    # segment_size din paper
                condition_in_question="after_condition",
                reorder_context="sort",
                dynamic_context_compression_ratio=0.3, # δτ din paper
                condition_compare=True,
                token_budget_ratio=1.4,
            )
        
        compressed_text = result["compressed_prompt"]
        compressed_tokens = count_tokens(compressed_text)
        
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000.0
        
        return compressed_text, original_tokens, compressed_tokens, latency_ms
