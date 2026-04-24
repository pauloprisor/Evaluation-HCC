# db/operations.py
import sqlite3
import json
from typing import Dict, Any, List
from .connection import get_connection

def create_experiment(method_name: str, task_name: str, token_budget: int, target_llm: str) -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        # Check if experiment exists
        cursor.execute(
            "SELECT id FROM experiments WHERE method_name=? AND task_name=? AND token_budget=? AND target_llm=?",
            (method_name, task_name, token_budget, target_llm)
        )
        row = cursor.fetchone()
        if row:
            return row['id']
            
        cursor.execute(
            """
            INSERT INTO experiments (method_name, task_name, token_budget, target_llm)
            VALUES (?, ?, ?, ?)
            """,
            (method_name, task_name, token_budget, target_llm)
        )
        conn.commit()
        return cursor.lastrowid

def is_compression_done(experiment_id: int, sample_id: str) -> bool:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM compressions WHERE experiment_id = ? AND sample_id = ?",
            (experiment_id, sample_id)
        )
        return cursor.fetchone() is not None

def save_compression(
    experiment_id: int, 
    sample_id: str, 
    task_type: str,
    question: str,
    ground_truth: str,
    all_classes: Any,
    original_tokens: int,
    compressed_tokens: int,
    compressed_context: str,
    metric_name: str,
    compression_latency_ms: int
) -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO compressions (
                experiment_id, sample_id, task_type, question,
                ground_truth, all_classes, original_tokens,
                compressed_tokens, compressed_context, metric_name,
                compression_latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id, sample_id, task_type, question,
                ground_truth, json.dumps(all_classes) if all_classes else None,
                original_tokens, compressed_tokens, compressed_context,
                metric_name, compression_latency_ms
            )
        )
        conn.commit()
        return cursor.lastrowid

def get_pending_compressions(experiment_id: int) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        # Returns compressions that don't have a corresponding result
        cursor.execute(
            """
            SELECT c.* FROM compressions c
            LEFT JOIN results r ON (c.experiment_id = r.experiment_id AND c.sample_id = r.sample_id)
            WHERE c.experiment_id = ? AND r.id IS NULL
            """,
            (experiment_id,)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def is_sample_done(experiment_id: int, sample_id: str) -> bool:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM results WHERE experiment_id = ? AND sample_id = ?",
            (experiment_id, sample_id)
        )
        return cursor.fetchone() is not None

def save_result(experiment_id: int, sample_id: str, llm_response: str, score: float, latency_ms: int) -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO results (experiment_id, sample_id, llm_response, score, latency_ms)
            VALUES (?, ?, ?, ?, ?)
            """,
            (experiment_id, sample_id, llm_response, score, latency_ms)
        )
        conn.commit()
        return cursor.lastrowid

def get_aggregated_results(method_name: str, token_budget: int) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT task_name, COUNT(*) as n_samples, AVG(score) as avg_score,
                   AVG(CAST(original_tokens AS FLOAT)/compressed_tokens) as avg_ratio,
                   AVG(compression_latency_ms) as avg_comp_lat,
                   AVG(latency_ms) as avg_llm_lat
            FROM v_aggregated
            WHERE method_name = ? AND token_budget = ?
            GROUP BY task_name
            """,
            (method_name, token_budget)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
