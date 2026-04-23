import time
from typing import Tuple
from .base import BaseMethod, count_tokens

class BaselineMethod(BaseMethod):
    def __init__(self, **kwargs):
        pass

    def get_name(self) -> str:
        return "baseline"

    def compress(self, context: str, question: str, token_budget: int) -> Tuple[str, int, int, float]:
        t0 = time.time()
        tokens = count_tokens(context)
        return context, tokens, tokens, 0.0
