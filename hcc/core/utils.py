import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    import logging
    try:
        nltk.download('punkt_tab')
        nltk.download('punkt')
    except Exception as e:
        logging.warning("NLTK not installed")

def split_sentences(raw_context:str) -> list[str]:
    if not str(raw_context).strip():
        return []
    
    parts = sent_tokenize(raw_context)
    sentences = []
    
    # Hard-limit chunks to prevent N^2 explosion in CrossEncoder math
    MAX_WORDS = 50
    
    for p in parts:
        p = p.strip()
        if len(p) <= 5: 
            continue
            
        words = p.split()
        if len(words) > MAX_WORDS:
            # Chunk the mutant sentence into equal, safe-sized pieces
            for i in range(0, len(words), MAX_WORDS):
                chunk = " ".join(words[i:i+MAX_WORDS])
                if len(chunk) > 5:
                    sentences.append(chunk)
        else:
            sentences.append(p)
            
    return sentences

def count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return max(1, len(text) // 4)
