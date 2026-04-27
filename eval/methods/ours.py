import os
import sys
import re
import time
import numpy as np
import warnings
import xgboost as xgb

from typing import Tuple

from .base import BaseMethod, count_tokens

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from hcc.core.utils import split_sentences
from hcc.core.cpc_base import ContextAwareScore
from hcc.core.llmlingua import LLMLingua


class OursCompressor(BaseMethod):
    def __init__(self, **kwargs):
        print("[Ours Setup] Loading XGBoost Model...")
        self.meta_classifier = xgb.XGBClassifier()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        xgb_path = os.path.abspath(os.path.join(script_dir, "..", "..", "hcc", "data", "xgboost_fusion.json"))
        
        try:
            self.meta_classifier.load_model(xgb_path)
            self.xgb_loaded = True
        except Exception:
            self.xgb_loaded = False
            warnings.warn(f"XGBoost model not loaded from {xgb_path}!")

        print("[Ours Setup] Loading BGE Reranker...")
        self.scorer = ContextAwareScore(model_name="BAAI/bge-reranker-base")

        print("[Ours Setup] Loading LLMLingua-2...")
        self.generative = LLMLingua(target_ratio=0.5)
        
    def get_name(self) -> str:
        return "ours"

    def _normalize_text(self, text: str) -> str:
        return re.sub(r'\W+', ' ', text).lower().strip()

    def compress(
        self,
        context: str,
        question: str,
        token_budget: int
    ) -> Tuple[str, int, int, float]:
        
        t0 = time.time()
        sentences = split_sentences(context)
        original_tokens = count_tokens(context)
        
        if not sentences or not self.xgb_loaded or original_tokens <= token_budget:
            t1 = time.time()
            return context, original_tokens, original_tokens, (t1 - t0) * 1000.0

        tfidf_scores = self.scorer.score_tfidf(question, sentences)
        bge_scores = self.scorer.score_bge(question, sentences)

        sorted_tfidf = sorted([(sc, idx) for idx, sc in enumerate(tfidf_scores)], reverse=True)
        sorted_bge = sorted([(sc, idx) for idx, sc in enumerate(bge_scores)], reverse=True)

       
        candidate_indices = list(range(len(sentences)))

        XGB_features = []
        normalized_query_words = self._normalize_text(question).split()
        query_length = len(normalized_query_words)

        for i in candidate_indices:
            normalized_sentence_words = self._normalize_text(sentences[i]).split()
            s_len = len(normalized_sentence_words)

            overlap_c = sum(1 for w in normalized_query_words if w in normalized_sentence_words)
            q_overlap_ratio = round(overlap_c / query_length, 4) if query_length > 0 else 0.0

            MAX_RANK = 15
            PENALTY = 16
            rank_tf, rank_bge = PENALTY, PENALTY    
            
            for rank, (score, idx) in enumerate(sorted_tfidf[:MAX_RANK]):
                if idx == i: rank_tf = rank + 1; break
            for rank, (score, idx) in enumerate(sorted_bge[:MAX_RANK]):
                if idx == i: rank_bge = rank + 1; break
            
            is_overlap = 1 if (rank_tf <= MAX_RANK and rank_bge <= MAX_RANK) else 0
            row = [tfidf_scores[i], bge_scores[i], s_len, is_overlap, query_length, rank_tf, rank_bge, q_overlap_ratio]
            XGB_features.append(row)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xgb_scores = self.meta_classifier.predict_proba(np.array(XGB_features))[:, 1]

        scored_candidates = []
        for list_idx, score in enumerate(xgb_scores):
            scored_candidates.append( (score, candidate_indices[list_idx]) )
            
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        selected_indices = set()
        intermediate_token_count = 0
        intermediate_budget = 2 * token_budget
        
        for score, original_idx in scored_candidates:
            sent_token_len = count_tokens(sentences[original_idx])
            if intermediate_token_count + sent_token_len > intermediate_budget:
                continue
                
            selected_indices.add(original_idx)
            intermediate_token_count += sent_token_len

        if not selected_indices:
            selected_indices.add(sorted_bge[0][1])

        chronological_indices = sorted(list(selected_indices))
        kept_sentences = [sentences[idx] for idx in chronological_indices]
        
        macro_filtered_context = " ".join(kept_sentences)
        macro_tokens = count_tokens(macro_filtered_context)
        
        if macro_tokens <= token_budget:
            t1 = time.time()
            return macro_filtered_context, original_tokens, macro_tokens, (t1 - t0) * 1000.0


        target_ratio = min(1.0, float(token_budget) / float(macro_tokens))
        self.generative.target_ratio = target_ratio
        
        result = self.generative.compress(macro_filtered_context)
        
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000.0
        
        return result.compressed_text, original_tokens, result.compressed_tokens, latency_ms
