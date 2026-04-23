import time
import os
from openai import OpenAI
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import LLM_BASE_URL, LLM_MODEL 
_llm_client = None

def get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        api_key = os.getenv("LLM_API_KEY", "ollama")
        _llm_client = OpenAI(base_url=LLM_BASE_URL, api_key=api_key)
    return _llm_client


PROMPT_TEMPLATE = """\
Answer the question ONLY based on the context below. Be concise.

Context:
{context}

Question: {question}

Answer:"""

def build_prompt(question: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(question=question, context=context)

def call_llm(prompt: str) -> tuple[str, int, int, int]:
    client = get_llm_client()
    t0 = time.time()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    latency_ms = int((time.time() - t0) * 1000)
    choice = resp.choices[0]
    text = choice.message.content or ""
    pt = resp.usage.prompt_tokens if resp.usage else 0
    ct = resp.usage.completion_tokens if resp.usage else 0
    
    return text.strip(), pt, ct, latency_ms
