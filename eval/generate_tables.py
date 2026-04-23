import sys
import os
import csv
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from db.connection import get_connection

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TASK_GROUPS = {
    "SingleDoc": ["narrativeqa", "qasper", "multifieldqa_en"],
    "MultiDoc":  ["hotpotqa", "2wikimqa", "musique"],
    "Summ.":     ["gov_report", "qmsum", "multi_news"],
    "FewShot":   ["trec", "triviaqa", "samsum"],
    "Synth.":    ["passage_count", "passage_retrieval_en"],
    "Code":      ["lcc", "repobench-p"],
}
GROUP_ORDER = ["SingleDoc", "MultiDoc", "Summ.", "FewShot", "Synth.", "Code"]
ALL_TASKS = [t for g in GROUP_ORDER for t in TASK_GROUPS[g]]

METHOD_ORDER = ["ours", "llmlingua2", "longllmlingua", "llmlingua", "selective_context", "cpc"]


def fetch_aggregated(token_budget: int) -> dict:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT method, task_type, avg_score, avg_compression_ratio,
                   avg_compression_latency_ms, avg_llm_latency_ms
            FROM v_aggregated WHERE token_budget = ?
            """,
            (token_budget,)
        )
        rows = cursor.fetchall()
    data = {}
    for row in rows:
        m, t = row["method"], row["task_type"]
        if m not in data:
            data[m] = {}
        data[m][t] = {
            "score": row["avg_score"] or 0.0,
            "ratio": row["avg_compression_ratio"] or 0.0,
            "comp_ms": row["avg_compression_latency_ms"] or 0.0,
            "llm_ms":  row["avg_llm_latency_ms"] or 0.0,
        }
    return data


def fetch_token_stats(token_budget: int) -> dict:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT e.method,
                   AVG(r.compressed_tokens) AS avg_tokens,
                   AVG(CAST(r.original_tokens AS FLOAT) / r.compressed_tokens) AS avg_ratio
            FROM results r
            JOIN experiments e ON r.experiment_id = e.id
            WHERE e.token_budget = ?
            GROUP BY e.method
            """,
            (token_budget,)
        )
        rows = cursor.fetchall()
    return {r["method"]: {"tokens": r["avg_tokens"] or 0, "ratio": r["avg_ratio"] or 0} for r in rows}


def ordered_methods(data):
    return [m for m in METHOD_ORDER if m in data] + [m for m in data if m not in METHOD_ORDER]


# ─── Tabel 1a: task cu task individual ───────────────────────────────────────
def build_table_individual(data, token_stats) -> list[dict]:
    rows = []
    for method in ordered_methods(data):
        td = data.get(method, {})
        ts = token_stats.get(method, {})
        row = {"Method": method}
        scores = []
        for task in ALL_TASKS:
            s = td.get(task, {}).get("score")
            if s is not None:
                row[task] = f"{s:.1f}"
                scores.append(s)
            else:
                row[task] = "-"
        row["AVG"]    = f"{sum(scores)/len(scores):.1f}" if scores else "-"
        row["Tokens"] = f"{int(ts['tokens'])}" if ts.get("tokens") else "-"
        row["1/τ"]    = f"{ts['ratio']:.1f}x" if ts.get("ratio") else "-"
        rows.append(row)
    return rows


# ─── Tabel 1b: grupat pe categorii ───────────────────────────────────────────
def build_table_grouped(data, token_stats) -> list[dict]:
    rows = []
    for method in ordered_methods(data):
        td = data.get(method, {})
        ts = token_stats.get(method, {})
        row = {"Method": method}
        all_scores = []
        for group in GROUP_ORDER:
            gs = [td.get(t, {}).get("score") for t in TASK_GROUPS[group]
                  if td.get(t, {}).get("score") is not None]
            if gs:
                avg = sum(gs) / len(gs)
                row[group] = f"{avg:.1f}"
                all_scores.append(avg)
            else:
                row[group] = "-"
        row["AVG"]    = f"{sum(all_scores)/len(all_scores):.1f}" if all_scores else "-"
        row["Tokens"] = f"{int(ts['tokens'])}" if ts.get("tokens") else "-"
        row["1/τ"]    = f"{ts['ratio']:.1f}x" if ts.get("ratio") else "-"
        rows.append(row)
    return rows


# ─── Tabel 2: latenta ────────────────────────────────────────────────────────
def build_table_latency(data) -> list[dict]:
    latencies = {}
    for method in ordered_methods(data):
        td = data.get(method, {})
        comp, llm = [], []
        for task in ALL_TASKS:
            if td.get(task, {}).get("comp_ms"):
                comp.append(td[task]["comp_ms"])
            if td.get(task, {}).get("llm_ms"):
                llm.append(td[task]["llm_ms"])
        avg_c = sum(comp)/len(comp) if comp else 0
        avg_l = sum(llm)/len(llm) if llm else 0
        latencies[method] = {"comp": avg_c, "llm": avg_l, "total": avg_c + avg_l}

    ours_total = latencies.get("ours", {}).get("total")
    rows = []
    for method in ordered_methods(data):
        lat = latencies[method]
        speedup = f"{lat['total']/ours_total:.2f}x" if ours_total else "-"
        rows.append({
            "Method":           method,
            "Compression (ms)": f"{lat['comp']:.0f}",
            "LLM (ms)":         f"{lat['llm']:.0f}",
            "Total (ms)":       f"{lat['total']:.0f}",
            "vs Ours":          speedup,
        })
    return rows


def write_csv(rows, path):
    if not rows:
        print(f"  [WARN] No data -> {path}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=rows[0].keys()).writeheader()
        csv.DictWriter(f, fieldnames=rows[0].keys()).writerows(rows)
    # Rewrire properly
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


def write_markdown(rows, path, title=""):
    if not rows:
        return
    cols = list(rows[0].keys())
    lines = []
    if title:
        lines.append(f"### {title}\n")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "-")) for c in cols) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_budget", type=int, default=None)
    args = parser.parse_args()

    budgets = [args.token_budget] if args.token_budget else [2000, 3000]

    for budget in budgets:
        print(f"\n--- token_budget={budget} ---")
        data = fetch_aggregated(budget)
        if not data:
            print("  [SKIP] No data in DB for this budget.")
            continue
        ts = fetch_token_stats(budget)

        # 1a - individual tasks
        t1a = build_table_individual(data, ts)
        write_csv(t1a,      os.path.join(RESULTS_DIR, f"table1a_individual_{budget}.csv"))
        write_markdown(t1a, os.path.join(RESULTS_DIR, f"table1a_individual_{budget}.md"),
                       title=f"All Tasks — {budget}-token constraint")

        # 1b - grouped
        t1b = build_table_grouped(data, ts)
        write_csv(t1b,      os.path.join(RESULTS_DIR, f"table1b_grouped_{budget}.csv"))
        write_markdown(t1b, os.path.join(RESULTS_DIR, f"table1b_grouped_{budget}.md"),
                       title=f"Grouped — {budget}-token constraint")

        # 2 - latency
        t2 = build_table_latency(data)
        write_csv(t2,      os.path.join(RESULTS_DIR, f"table2_latency_{budget}.csv"))
        write_markdown(t2, os.path.join(RESULTS_DIR, f"table2_latency_{budget}.md"),
                       title=f"Latency — {budget}-token constraint")

    print("\nDone. Files in results/")


if __name__ == "__main__":
    main()
