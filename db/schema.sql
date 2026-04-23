CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    method TEXT NOT NULL,
    dataset TEXT NOT NULL,
    token_budget INTEGER NOT NULL,
    target_llm TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER REFERENCES experiments(id),
    sample_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    original_context TEXT NOT NULL,
    compressed_context TEXT,
    question TEXT NOT NULL,
    ground_truth TEXT NOT NULL,
    llm_response TEXT,
    original_tokens INTEGER NOT NULL,
    compressed_tokens INTEGER,
    metric_name TEXT NOT NULL,
    score FLOAT NOT NULL,
    compression_latency_ms INTEGER,
    llm_latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_results_experiment 
    ON results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_results_task 
    ON results(task_type);

CREATE VIEW IF NOT EXISTS v_aggregated AS
SELECT
    e.method,
    e.dataset,
    e.token_budget,
    r.task_type,
    r.metric_name,
    COUNT(*) AS n_samples,
    ROUND(AVG(r.score), 2) AS avg_score,
    CAST(AVG(r.original_tokens) AS FLOAT) /
        AVG(r.compressed_tokens) AS avg_compression_ratio,
    AVG(r.compression_latency_ms) AS avg_compression_latency_ms,
    AVG(r.llm_latency_ms) AS avg_llm_latency_ms
FROM experiments e
JOIN results r ON r.experiment_id = e.id
GROUP BY e.method, e.dataset, e.token_budget, r.task_type, r.metric_name;
