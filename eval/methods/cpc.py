import sys
import os
import time
import torch
import warnings
from typing import Tuple

from .base import BaseMethod, count_tokens

# Înregistrăm dinamic folderul 'cpc_repo' ca modul Python pentru ca librăria să își poată apela intern modulele ('model', 'util', etc)
cpc_repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'cpc_repo')
if os.path.abspath(cpc_repo_path) not in sys.path:
    sys.path.insert(0, os.path.abspath(cpc_repo_path))

try:
    from prompt_compressor import PromptCompressorCPC
    from model.common import ModelType
except ImportError as e:
    PromptCompressorCPC = None
    ModelType = None
    # Lăsăm eroarea liberă pentru momentul inițializării


class CPCCompressorMethod(BaseMethod):
    def __init__(self, **kwargs):
        if PromptCompressorCPC is None:
            raise ImportError("CPC repository or its internal dependencies could not be loaded. Ensure it sits in Evaluation-HCC/cpc_repo")
            
        print("[CPC] Setup: Loading CPC architecture (LLaMA-1B)...")
    
        self.compressor = PromptCompressorCPC(
            model_type=ModelType.LLAMA,
            use_question_as_suffix=False,
            use_openai_tokenizer_to_measure_length=True
        )

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
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compressed_text = self.compressor.compress(
                context=context,
                question=question,
                compression_target_tokens=token_budget,
            )
            
        compressed_tokens = count_tokens(compressed_text)
        
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000.0
        
        return compressed_text, original_tokens, compressed_tokens, latency_ms
