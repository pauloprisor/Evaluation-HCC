import re
import string
import collections

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def contains_answer(pred: str, aliases: list) -> bool:
    pred_norm = normalize(pred)
    return any(normalize(alias) in pred_norm for alias in aliases)

def token_f1(pred: str, truth: str) -> float:
    pred_tokens  = normalize(pred).split()
    truth_tokens = normalize(truth).split()
    
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
        
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall    = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def f1_over_aliases(pred: str, aliases: list) -> float:
    if not aliases:
        return 0.0
    return max(token_f1(pred, alias) for alias in aliases)

_bertscore_fn = None

def compute_bert_score(pred: str, truth: str) -> float | None:
    global _bertscore_fn
    try:
        if _bertscore_fn is None:
            import logging
            logging.getLogger("transformers").setLevel(logging.ERROR)
            from bert_score import BERTScorer
            _bertscore_fn = BERTScorer(
                model_type="distilbert-base-uncased",
                num_layers=5,
                all_layers=False,
                idf=False,
                device=None,
            )
        P, R, F = _bertscore_fn.score([pred], [truth])
        return float(F[0])
    except Exception as e:
        print(f"Error computing BERTScore: {e}")
        return None
