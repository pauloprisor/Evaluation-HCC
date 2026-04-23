import time
import torch
from .models import CompressionResult
from .utils import count_tokens

try:
    from llmlingua import PromptCompressor
except ImportError:
    PromptCompressor = None

class LLMLingua:
    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        target_ratio: float = 0.5,
    ):
        if PromptCompressor is None:
            raise ImportError("llmlingua not installed")
        
        self.target_ratio = target_ratio
        
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        self.compressor = PromptCompressor(
            model_name=model_name,
            use_llmlingua2 = True,
            device_map = self.device
        )

    def compress(self, context_text: str) -> CompressionResult:
        t0 = time.time()

        original_tokens = count_tokens(context_text)

        kwargs = dict(
            context = context_text,
            rate = self.target_ratio,
        )
        result = self.compressor.compress_prompt(**kwargs)
        compressed_text= result["compressed_prompt"]

        compressed_tokens = count_tokens(compressed_text)
        latency_ms = int((time.time()-t0) *1000)

        ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 1.0

        return CompressionResult(
            compressed_text = compressed_text,
            original_tokens= original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio= ratio,
            latency_ms= latency_ms
        )

