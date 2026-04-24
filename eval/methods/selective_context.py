import time
from typing import Tuple

from .base import BaseMethod, count_tokens

try:
    from selective_context import SelectiveContext
except ImportError:
    SelectiveContext = None


class SelectiveContextMethod(BaseMethod):
    def __init__(self, **kwargs):
        if SelectiveContext is None:
            raise ImportError("Please install selective-context package.")
        
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        self.sc = SelectiveContext(
            model_type='deepseek-ai/deepseek-llm-7b-base', 
            lang='en',
            device=device
        )
        self._tokenizer = self.sc.tokenizer

    
    def compress(
        self, 
        context: str, 
        question: str, 
        token_budget: int
    ) -> Tuple[str, int, int, float]:

        start = time.time()
        original_tokens = count_tokens(context)
        
        ratio = self._compute_ratio(context, token_budget)

        try:
            compressed, _ = self.sc(context, reduce_ratio=ratio)
        except Exception as e:
            print(f"[SelectiveContext] Error: {e}")
            compressed = context
        
        latency = (time.time() - start) * 1000.0
        
        return (
            compressed,
            original_tokens,
            count_tokens(compressed),
            latency
        )
    
    def _compute_ratio(self, context: str, token_budget: int) -> float:
        original = count_tokens(context)
        if original <= token_budget:
            return 0.0
        return 1.0 - (token_budget / original)
    
    def get_name(self) -> str:
        return "selective_context"
