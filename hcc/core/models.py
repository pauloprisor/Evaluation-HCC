from dataclasses import dataclass

@dataclass
class CompressionResult:
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    latency_ms: int


    