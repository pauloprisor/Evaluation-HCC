import time
import torch
import warnings
from typing import Tuple

from .base import BaseMethod, count_tokens

try:
    from llmlingua import PromptCompressor
except ImportError:
    PromptCompressor = None

class LLMLingua2Compressor(BaseMethod):
    def __init__(self, **kwargs):
        if PromptCompressor is None:
            raise ImportError("Please install llmlingua package first.")
        
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        hf_bert_id = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
        
        print(f"[LLMLingua-2] Setup: Loading Token Classification Model {hf_bert_id} pe {self.device}...")
        self.compressor = PromptCompressor(
            model_name=hf_bert_id,
            use_llmlingua2=True,
            device_map=self.device
        )

    def get_name(self) -> str:
        return "llmlingua2"

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
            # LLMLingua-2 aruncă textul direct prin BERT token classify
            result = self.compressor.compress_prompt(
                context=context,
                rate=target_ratio,
            )
        
        compressed_text = result["compressed_prompt"]
        compressed_tokens = count_tokens(compressed_text)
        
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000.0
        
        return compressed_text, original_tokens, compressed_tokens, latency_ms
