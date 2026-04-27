# Evaluation Repo for HCC paper

### Config and run
`chmod +x setup.sh`
`./setup.sh`
`source venv/bin/activate`
`python eval/run_evaluation.py`
`for b in 2000 3000; do python eval/run_evaluation.py --methods baseline ours llmlingua llmlingua2 cpc selective_context longllmlingua --token_budget $b; done`
`python -m eval.compress_all --methods all --tasks all --token_budget 1000`
`python -m eval.run_llm --methods all --tasks all --token_budget 1000`


### Note
- Should change the models for llmlingua1,longllmlingua,selective_context
- Should have Ollama running and need to change the Target LLM to the one we want to test from config file
- We can change token budget and tasks from config file.
- Need to add new datasets for more evaluation.

