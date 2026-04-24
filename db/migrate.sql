-- db/migrate.sql

CREATE TABLE IF NOT EXISTS compressions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER REFERENCES experiments(id),
    sample_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    question TEXT NOT NULL,
    ground_truth TEXT NOT NULL,
    all_classes TEXT,
    original_tokens INTEGER NOT NULL,
    compressed_tokens INTEGER,
    compressed_context TEXT,
    metric_name TEXT NOT NULL,
    compression_latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

DROP VIEW IF EXISTS v_aggregated;
CREATE VIEW v_aggregated AS
SELECT 
    e.id as experiment_id,
    e.method_name,
    e.task_name,
    e.token_budget,
    c.sample_id,
    c.original_tokens,
    c.compressed_tokens,
    r.llm_response,
    r.score,
    r.latency_ms as llm_latency,
    c.compression_latency_ms
FROM experiments e
JOIN compressions c ON e.id = c.experiment_id
LEFT JOIN results r ON (e.id = r.experiment_id AND c.sample_id = r.sample_id);
