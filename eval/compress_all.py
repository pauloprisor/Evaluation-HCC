# eval/compress_all.py
import argparse
import yaml
import os
import gc
import torch
from tqdm import tqdm
from typing import List

from eval.run_evaluation import (
    METHOD_REGISTRY, 
    DATASETS, 
    DATASET2METRIC, 
    load_task_data,
    count_tokens
)
from db.operations import create_experiment, is_compression_done, save_compression

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Compress contexts and save to DB")
    parser.add_argument("--methods", nargs="+", help="Methods to run (or 'all')")
    parser.add_argument("--tasks", nargs="+", help="Tasks to run (or 'all')")
    parser.add_argument("--token_budget", type=int, default=2000, help="Target token budget")
    parser.add_argument("--limit", type=int, help="Limit samples per task")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve methods
    available_methods = list(METHOD_REGISTRY.keys())
    if not args.methods or "all" in args.methods:
        methods_to_run = config.get("methods", available_methods)
    else:
        methods_to_run = args.methods

    # Resolve tasks
    if not args.tasks or "all" in args.tasks:
        tasks_to_run = config.get("datasets", DATASETS)
    else:
        tasks_to_run = args.tasks

    token_budget = args.token_budget
    target_llm = config.get("target_llm", "llama3")

    print(f"🚀 Starting Stage 1: Compression Only")
    print(f"Methods: {methods_to_run}")
    print(f"Tasks: {tasks_to_run}")
    print(f"Budget: {token_budget}")
    print("-" * 30)

    for method_name in methods_to_run:
        print(f"\n📦 Loading Method: {method_name}")
        method_class = METHOD_REGISTRY.get(method_name)
        if not method_class:
            print(f"Skip: Method {method_name} not found in registry.")
            continue
            
        # Initialize method
        method_instance = method_class()

        for task in tasks_to_run:
            print(f"  Task: {task}")
            exp_id = create_experiment(method_name, task, token_budget, target_llm)
            
            data = load_task_data(task)
            metric_name = DATASET2METRIC.get(task, "f1")
            
            samples = list(data)
            if args.limit:
                samples = samples[:args.limit]

            for idx, sample in enumerate(tqdm(samples, desc=f"Compressing {task}")):
                sample_id = f"{task}_{idx}"
                
                if is_compression_done(exp_id, sample_id):
                    continue

                context = sample["context"]
                question = sample["input"]
                ground_truth = sample["answers"][0] if sample.get("answers") else ""
                all_classes = sample.get("all_classes")
                
                orig_tokens = count_tokens(context)
                
                try:
                    compressed_ctx, _, comp_tokens, comp_lat = method_instance.compress(
                        context=context,
                        question=question,
                        token_budget=token_budget
                    )
                    
                    save_compression(
                        experiment_id=exp_id,
                        sample_id=sample_id,
                        task_type=task,
                        question=question,
                        ground_truth=ground_truth,
                        all_classes=all_classes,
                        original_tokens=orig_tokens,
                        compressed_tokens=comp_tokens,
                        compressed_context=compressed_ctx,
                        metric_name=metric_name,
                        compression_latency_ms=comp_lat
                    )
                except Exception as e:
                    print(f"\n[ERROR] Compression failed for {sample_id}: {e}")

        # Cleanup after each method to free VRAM
        print(f"🧹 Unloading {method_name} and clearing GPU cache...")
        del method_instance
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_all_cache() if hasattr(torch.cuda, 'empty_all_cache') else torch.cuda.empty_cache()

    print("\n✅ Stage 1 Complete. Check 'compressions' table in DB.")

if __name__ == "__main__":
    main()
