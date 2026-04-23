from abc import ABC, abstractmethod

def count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return max(1, len(text) // 4)

class BaseMethod(ABC):
    @abstractmethod
    def compress(
        self,
        context: str,
        question: str,
        token_budget: int
    ) -> tuple[str, int, int, float]:
        """
        Return:
        - compressed_context: str
        - original_tokens: int
        - compressed_tokens: int  
        - compression_latency_ms: float
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError
