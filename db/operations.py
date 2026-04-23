import sqlite3
from typing import Dict, Any, List
from .connection import get_connection

def create_experiment(name: str, method: str, dataset: str, token_budget: int, target_llm: str) -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO experiments (name, method, dataset, token_budget, target_llm)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, method, dataset, token_budget, target_llm)
        )
        conn.commit()
        return cursor.lastrowid

def save_result(experiment_id: int, sample_id: str, task_type: str,
               original_context: str, compressed_context: str,
               question: str, ground_truth: str, llm_response: str,
               original_tokens: int, compressed_tokens: int,
               metric_name: str, score: float,
               compression_latency_ms: int, llm_latency_ms: int) -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO results (
                experiment_id, sample_id, task_type,
                original_context, compressed_context, question,
                ground_truth, llm_response, original_tokens,
                compressed_tokens, metric_name, score,
                compression_latency_ms, llm_latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id, sample_id, task_type,
                original_context, compressed_context, question,
                ground_truth, llm_response, original_tokens,
                compressed_tokens, metric_name, score,
                compression_latency_ms, llm_latency_ms
            )
        )
        conn.commit()
        return cursor.lastrowid

def get_aggregated_results(method: str, token_budget: int) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT task_type, metric_name, n_samples, avg_score,
                   avg_compression_ratio, avg_compression_latency_ms, avg_llm_latency_ms
            FROM v_aggregated
            WHERE method = ? AND token_budget = ?
            ORDER BY task_type
            """,
            (method, token_budget)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def experiment_exists(method: str, dataset: str, token_budget: int) -> bool:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id FROM experiments
            WHERE method = ? AND dataset = ? AND token_budget = ?
            """,
            (method, dataset, token_budget)
        )
        return cursor.fetchone() is not None

def is_sample_done(experiment_id: int, sample_id: str) -> bool:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id FROM results
            WHERE experiment_id = ? AND sample_id = ?
            """,
            (experiment_id, sample_id)
        )
        return cursor.fetchone() is not None
