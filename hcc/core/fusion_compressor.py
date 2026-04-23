import os
import re
import numpy as np 
import warnings
import xgboost as xgb
import time

from .models import CompressionResult
from .utils import split_sentences, count_tokens
from .cpc_base import ContextAwareScore
from .llmlingua import LLMLingua

class HybridContextCompressor:
    def __init__(self,target_ratio: float = 0.5):
        self.scorer = ContextAwareScore(model_name="BAAI/bge-reranker-base")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        xgb_model_path = os.path.abspath(os.path.join(script_dir,"..","data","xgboost_fusion.json"))

        self.meta_classifier = xgb.XGBClassifier()
        try:
            self.meta_classifier.load_model(xgb_model_path)
            self.xgb_loaded = True
        except Exception:
            self.xgb_loaded = False
            warnings.warn("xgb not loaded")
        
        self.xgb_treshold = 0.2

        self.generative = LLMLingua(target_ratio=target_ratio)
    
    def _normalize_text(self, text:str)->str:
        return re.sub(r'\W+',' ',text).lower().strip()

    # Fusioned compression
    def compress(self,context:str, query:str) -> CompressionResult:
        t0=time.time()

        sentences = split_sentences(context)
        t_split = time.time()
        print(f"[HCC Profile] Split Sentences: {len(sentences)} sentences | Time: {t_split - t0:.3f}s")
        
        if not sentences or not self.xgb_loaded:
            return self._empty_result(context,t0)
        
        tfidf_scores = self.scorer.score_tfidf(query, sentences)
        t_tfidf = time.time()
        print(f"[HCC Profile] TF-IDF scoring: {t_tfidf - t_split:.3f}s")
        
        bge_scores = self.scorer.score_bge(query, sentences)
        t_bge = time.time()
        print(f"[HCC Profile] BGE CrossEncoder: {t_bge - t_tfidf:.3f}s")

        sorted_tfidf = sorted([(sc,idx) for idx,sc in enumerate(tfidf_scores)], reverse=True)
        sorted_bge = sorted([(sc,idx) for idx,sc in enumerate(bge_scores)], reverse=True)

        candidate_indices = set()
        for i in range(len(sentences)):
            if bge_scores[i] >=0.15 or tfidf_scores[i] >=0.10:
                candidate_indices.add(i)

        #Keep the best sentence of both methods
        if sorted_tfidf: candidate_indices.add(sorted_tfidf[0][1])
        if sorted_bge: candidate_indices.add(sorted_bge[0][1])
        candidate_indices = sorted(list(candidate_indices))
        if not candidate_indices:
            return self._empty_result(context,t0)
        
        # XGB Feature assembly
        XGB_features = []
        normalized_query_words = self._normalize_text(query).split()
        query_length = len(normalized_query_words)

        for i in candidate_indices:
            normalized_sentence_words = self._normalize_text(sentences[i]).split()
            s_len = len(normalized_sentence_words)

            # Overlap Query with Sentence
            overlap_c = sum(1 for w in normalized_query_words if w in normalized_sentence_words)
            q_overlap_ratio = round(overlap_c / query_length, 4) if query_length > 0 else 0.0

            # Sentence rank
            MAX_RANK = 15
            PENALTY = 16

            rank_tf, rank_bge = PENALTY, PENALTY    
            for rank,(score,idx) in enumerate(sorted_tfidf[:MAX_RANK]):
                if idx == i: rank_tf = rank+1; break
            for rank, (score,idx) in enumerate(sorted_bge[:MAX_RANK]):
                if idx == i: rank_bge = rank+1; break
            
            # Both methods have the same sentence between the first 15 choices
            is_overlap = 1 if (rank_tf <=MAX_RANK and rank_bge <= MAX_RANK) else 0

            row = [tfidf_scores[i], bge_scores[i], s_len, is_overlap, query_length, rank_tf, rank_bge, q_overlap_ratio]
            XGB_features.append(row)

        t_xgb_prep = time.time()

        # Running XGB on the rows we got
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xgb_scores = self.meta_classifier.predict_proba(np.array(XGB_features))[:,1]

        t_xgb = time.time()
        print(f"[HCC Profile] XGBoost assembly & inference: {t_xgb - t_bge:.3f}s")

        # Assemble the filtered sentences
        kept_sentences = []
        for list_idx,score in enumerate(xgb_scores):
            if score >=self.xgb_treshold:
                original_idx = candidate_indices[list_idx]
                kept_sentences.append(sentences[original_idx])
        
        if not kept_sentences:
            kept_sentences.append(sentences[sorted_bge[0][1]])

        # Concatenating the sentences
        filtered_sentences = " ".join(kept_sentences)
        
        print(f"[HCC Profile] Pre-LLMLingua Text Length: {len(filtered_sentences.split())} words")
        
        #Running the LLMLingua2 on the filtered sentences
        result = self.generative.compress(filtered_sentences)
        
        t_llmlingua = time.time()
        print(f"[HCC Profile] LLMLingua-2 Compression: {t_llmlingua - t_xgb:.3f}s")
        print(f"[HCC Profile] --- Total Pipeline Time --- : {t_llmlingua - t0:.3f}s")

        # Parameters for compression output
        original_tokens = count_tokens(context)
        compressed_tokens = result.compressed_tokens

        return CompressionResult(
            compressed_text=result.compressed_text,
            original_tokens=original_tokens,
            compressed_tokens = compressed_tokens,
            compression_ratio = original_tokens/ compressed_tokens if compressed_tokens > 0 else 1.0,
            latency_ms= int((time.time()-t0) * 1000)
        )
    
    def _empty_result(self, context:str, t0:float) -> CompressionResult:
        ot = count_tokens(context)
        return CompressionResult(context, ot, ot, 1.0, int((time.time() - t0) * 1000))
            
        

