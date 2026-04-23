import re
import math
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

try:
    import torch
    if torch.cuda.is_available() :
        _DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        _DEVICE = "mps"
    else:
        _DEVICE = "cpu"
except ImportError:
    _DEVICE = "cpu"

print(f"Using device: {_DEVICE}")

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

class ContextAwareScore:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        if CrossEncoder is None:
            raise ImportError("sentence-transformers not installed.")
        self.model = CrossEncoder(model_name, max_length=512, device=_DEVICE)
    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[a-z0-9]+\b", text.lower())

    def _tfidf_vectors(self, sentences: list[str]) -> tuple[list[dict], dict]:
        tf_vectors = []
        df: dict[str, int] = {}
        for sent in sentences:
            tokens = self._tokenize(sent)
            if not tokens:
                tf_vectors.append({})
                continue
            counts: dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            tf = {t: c / len(tokens) for t, c in counts.items()}
            tf_vectors.append(tf)
            for t in tf:
                df[t] = df.get(t, 0) + 1
        N = len(sentences)
        idf = {t: math.log((N + 1) / (freq + 1)) for t, freq in df.items()}
        return tf_vectors, idf

    def _tfidf_for(self, tokens: list[str], idf: dict) -> dict:
        if not tokens: return {}
        counts: dict[str, int] = {}
        for t in tokens: counts[t] = counts.get(t, 0) + 1
        return {t: (c / len(tokens)) * idf.get(t, 0) for t, c in counts.items()}

    def _cosine(self, a: dict, b: dict) -> float:
        if not a or not b: return 0.0
        dot = sum(a.get(t, 0) * v for t, v in b.items())
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0: return 0.0
        return dot / (norm_a * norm_b)

    # Lexical score
    def score_tfidf(self, query: str, context_sentences: list[str]) -> list[float]:
        if not context_sentences or not query:
            return [0.0] * len(context_sentences)
        
        tf_vectors, idf = self._tfidf_vectors(context_sentences)
        q_tokens = self._tokenize(query)
        q_vec = self._tfidf_for(q_tokens, idf)
        return [self._cosine(tv, q_vec) for tv in tf_vectors]

    # Semantic Score
    def score_bge(self, query: str, context_sentences:list[str]) -> list[float]:
        if not context_sentences or not query:
            return [0.0] * len(context_sentences)
        
        pairs = [[query, sent] for sent in context_sentences]
        try:
            raw_scores = self.model.predict(pairs, batch_size=16)
        except RuntimeError:
            # OOM fallback: batch cu 1 sample
            raw_scores = self.model.predict(pairs, batch_size=1)

        mn,mx = min(raw_scores), max(raw_scores)

        if mn == mx:
            return [1.0] * len(context_sentences)
        
        normalized = [(s-mn)/(mx-mn) for s in raw_scores]

        return normalized
            

        

    
        