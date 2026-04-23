import sys
import os
import json
import argparse
import yaml
from tqdm import tqdm
import gc
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from db.connection import get_connection
from db.operations import (
    create_experiment,
    save_result,
    is_sample_done,
    experiment_exists,
)
from eval.llm import LlamaLLM

from eval.methods.baseline import BaselineMethod
from eval.methods.ours import OursCompressor
from eval.methods.llmlingua import LLMLingua1Compressor
from eval.methods.llmlingua2 import LLMLingua2Compressor
from eval.methods.longllmlingua import LongLLMLinguaCompressor
from eval.methods.selective_context import SelectiveContextMethod
from eval.methods.cpc import CPCCompressorMethod

from eval.methods.base import count_tokens

LONGBENCH_PATH = os.path.join(ROOT, "LongBench", "LongBench")
sys.path.insert(0, LONGBENCH_PATH)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "longbench_eval",
    os.path.join(LONGBENCH_PATH, "eval.py")
)
_lb_eval = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_lb_eval)
scorer = _lb_eval.scorer

with open(os.path.join(LONGBENCH_PATH, "config", "dataset2prompt.json")) as f:
    DATASET2PROMPT = json.load(f)

with open(os.path.join(LONGBENCH_PATH, "config", "dataset2maxlen.json")) as f:
    DATASET2MAXLEN = json.load(f)

DATASET2METRIC = {
    "narrativeqa": "f1",
    "qasper": "f1",
    "multifieldqa_en": "f1",
    "multifieldqa_zh": "rouge_zh",
    "hotpotqa": "f1",
    "2wikimqa": "f1",
    "musique": "f1",
    "dureader": "rouge_zh",
    "gov_report": "rouge",
    "qmsum": "rouge",
    "multi_news": "rouge",
    "vcsum": "rouge_zh",
    "trec": "classification",
    "triviaqa": "f1",
    "samsum": "rouge",
    "lsht": "classification",
    "passage_retrieval_en": "retrieval",
    "passage_count": "count",
    "passage_retrieval_zh": "retrieval_zh",
    "lcc": "code_sim",
    "repobench-p": "code_sim",
}

METHOD_REGISTRY = {
    "baseline": BaselineMethod,
    "ours": OursCompressor,
    "llmlingua": LLMLingua1Compressor,
    "llmlingua2": LLMLingua2Compressor,
    "longllmlingua": LongLLMLinguaCompressor,
    "selective_context": SelectiveContextMethod,
    "cpc": CPCCompressorMethod,
}


def parse_args():
    parser = argparse.ArgumentParser(description="HCC Evaluation Runner")
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Lista de metode (default: all din config.yaml)"
    )
    parser.add_argument(
        "--dataset", type=str, default="longbench",
        help="Dataset de evaluat (default: longbench)"
    )
    parser.add_argument(
        "--token_budget", type=int, default=3000,
        help="Token budget pentru compresie (default: 3000)"
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Taskuri specifice (default: all din config.yaml)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Numarul maxim de sample-uri per task (default: toate)"
    )
    return parser.parse_args()


def load_config():
    config_path = os.path.join(ROOT, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_experiment_id(method_name: str, task: str, token_budget: int, config: dict) -> int:
    from db.connection import get_connection
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM experiments WHERE method=? AND dataset=? AND token_budget=?",
            (method_name, task, token_budget)
        )
        row = cursor.fetchone()
        if row:
            return row["id"]

    exp_name = f"{method_name}_{task}_budget{token_budget}"
    return create_experiment(
        name=exp_name,
        method=method_name,
        dataset=task,
        token_budget=token_budget,
        target_llm=config["target_llm"],
    )


def run_task(method_instance, method_name: str, task: str,
             token_budget: int, limit, llm: LlamaLLM, config: dict):
    print(f"\n{'='*60}")
    print(f"Method: {method_name} | Task: {task} | Budget: {token_budget}")
    print(f"{'='*60}")

    data_path = os.path.join(ROOT, "data", f"{task}.jsonl")
    if not os.path.exists(data_path):
        print(f"[ERROR] Fisierul local {data_path} nu exista. Ruleaza download_longbench.py intai.")
        return
    with open(data_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    exp_id = get_experiment_id(method_name, task, token_budget, config)

    prompt_format = DATASET2PROMPT[task]
    max_new_tokens = DATASET2MAXLEN[task]
    metric_name = DATASET2METRIC[task]

    samples = list(data)
    if limit is not None:
        samples = samples[:limit]

    pending = []
    skipped = 0
    for idx, sample in enumerate(samples):
        sample_id = f"{task}_{idx}"
        if is_sample_done(exp_id, sample_id):
            skipped += 1
        else:
            pending.append((idx, sample, sample_id))

    if skipped > 0:
        print(f"  Skipped {skipped} already-done samples.")

    if not pending:
        return

    # Compress
    print(f"  [Phase 1/2] Compresie pe {len(pending)} sample-uri...")
    compressed_batch = []
    for idx, sample, sample_id in tqdm(pending, desc=f"[compress] {method_name}/{task}"):
        context = sample["context"]
        question = sample["input"]
        answers = sample["answers"]
        all_classes = sample.get("all_classes", None)

        actual_orig_tokens = count_tokens(context)

        try:
            compressed_ctx, _, comp_tokens, comp_lat = method_instance.compress(
                context=context,
                question=question,
                token_budget=token_budget,
            )
        except Exception as e:
            print(f"\n[WARN] compress() failed on sample {idx}: {e}")
            continue

        compressed_batch.append({
            "idx": idx,
            "sample_id": sample_id,
            "question": question,
            "answers": answers,
            "all_classes": all_classes,
            "compressed_ctx": compressed_ctx,
            "orig_tokens": actual_orig_tokens,
            "comp_tokens": comp_tokens,
            "comp_lat": comp_lat,
        })

    if not compressed_batch:
        return

    # LLM generate 
    print(f"  [Phase 2/2] LLM generate pe {len(compressed_batch)} sample-uri...")
    for item in tqdm(compressed_batch, desc=f"[llm] {method_name}/{task}"):
        prompt = prompt_format.format(context=item["compressed_ctx"], input=item["question"])

        response, _, llm_lat = llm.generate(prompt, max_new_tokens=max_new_tokens)

        score = scorer(
            dataset=task,
            predictions=[response],
            answers=[item["answers"]],
            all_classes=item["all_classes"],
        )

        save_result(
            experiment_id=exp_id,
            sample_id=item["sample_id"],
            task_type=task,
            original_context="",   
            compressed_context=item["compressed_ctx"],
            question=item["question"],
            ground_truth=str(item["answers"]),
            llm_response=response,
            original_tokens=item["orig_tokens"],
            compressed_tokens=item["comp_tokens"],
            metric_name=metric_name,
            score=score,
            compression_latency_ms=int(item["comp_lat"]),
            llm_latency_ms=int(llm_lat),
        )



def unload_method(method_instance, method_name: str):
    print(f"  Unloading {method_name} from memory...")
    del method_instance
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    args = parse_args()
    config = load_config()

    methods_to_run = args.methods or config["methods"]
    tasks_to_run = args.tasks or config["datasets"]["longbench"]["tasks"]

    llm = LlamaLLM(model_path=config["target_llm"])

    for method_name in methods_to_run:
        if method_name not in METHOD_REGISTRY:
            print(f"[WARN] Metoda '{method_name}' nu exista in registru. Skip.")
            continue

        print(f"\nLoading method: {method_name}...")
        try:
            method_instance = METHOD_REGISTRY[method_name]()
        except Exception as e:
            print(f"[SKIP] Metoda '{method_name}' nu a putut fi incarcata: {e}")
            continue

        for task in tasks_to_run:
            if task not in DATASET2PROMPT:
                print(f"[WARN] Task '{task}' necunoscut in LongBench. Skip.")
                continue

            try:
                run_task(
                    method_instance=method_instance,
                    method_name=method_name,
                    task=task,
                    token_budget=args.token_budget,
                    limit=args.limit,
                    llm=llm,
                    config=config,
                )
            except Exception as e:
                print(f"[ERROR] run_task crashed: {method_name}/{task}: {e}")

        unload_method(method_instance, method_name)

    print("\nEvaluare finalizata.")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
