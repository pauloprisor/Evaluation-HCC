# eval/run_llm.py
import argparse
import yaml
import json
from tqdm import tqdm
from typing import List

from eval.run_evaluation import (
    METHOD_REGISTRY, 
    DATASETS, 
    DATASET2MAXLEN,
    DATASET2PROMPT,
    scorer
)
from eval.llm import LlamaLLM
from db.operations import create_experiment, get_pending_compressions, is_sample_done, save_result

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Run LLM on compressed contexts and score")
    parser.add_argument("--methods", nargs="+", help="Methods to run (or 'all')")
    parser.add_argument("--tasks", nargs="+", help="Tasks to run (or 'all')")
    parser.add_argument("--token_budget", type=int, default=2000, help="Target token budget")
    parser.add_argument("--limit", type=int, help="Limit samples (not used, depends on compressions table)")
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
    target_llm_name = config.get("target_llm", "llama3")

    print(f"🤖 Starting Stage 2: LLM Evaluation")
    print(f"Loading LLM: {target_llm_name}")
    llm = LlamaLLM(target_llm_name)
    print("-" * 30)

    for method_name in methods_to_run:
        for task in tasks_to_run:
            exp_id = create_experiment(method_name, task, token_budget, target_llm_name)
            
            pending = get_pending_compressions(exp_id)
            if not pending:
                continue

            print(f"\n🚀 Running LLM for {method_name}/{task} ({len(pending)} samples)")
            
            # LongBench config for generation length
            max_new_tokens = DATASET2MAXLEN.get(task, 128)
            prompt_format = DATASET2PROMPT.get(task, "Context: {context}\n\nQuestion: {input}\n\nAnswer:")

            for comp in tqdm(pending, desc=f"LLM {task}"):
                sample_id = comp["sample_id"]
                
                # Double check resume
                if is_sample_done(exp_id, sample_id):
                    continue

                # Prepare prompt using official format
                try:
                    prompt = prompt_format.format(context=comp['compressed_context'], input=comp['question'])
                except Exception:
                    # Fallback if keys don't match exactly
                    prompt = f"Context: {comp['compressed_context']}\n\nQuestion: {comp['question']}\n\nAnswer:"
                
                try:
                    response, _, llm_lat = llm.generate(prompt, max_new_tokens=max_new_tokens)
                    
                    # Score
                    ground_truth = comp["ground_truth"]
                    all_classes = json.loads(comp["all_classes"]) if comp["all_classes"] else None
                    
                    s = scorer(task, response, ground_truth, all_classes)
                    
                    save_result(
                        experiment_id=exp_id,
                        sample_id=sample_id,
                        llm_response=response,
                        score=s,
                        latency_ms=int(llm_lat)
                    )
                except Exception as e:
                    print(f"\n[ERROR] LLM/Scoring failed for {sample_id}: {e}")

    print("\n✅ Stage 2 Complete. Results are in 'results' table.")

if __name__ == "__main__":
    main()
