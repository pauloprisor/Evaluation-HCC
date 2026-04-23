# Evaluation Repo for HCC paper

### Config and run
`chmod +x setup.sh`
`./setup.sh`
`source venv/bin/activate`
`python eval/run_evaluation.py`

### Note
- Should change the models for llmlingua1,longllmlingua,selective_context
- Should have Ollama running and need to change the Target LLM to the one we want to test from config file
- We can change token budget and tasks from config file.
- Need to add new datasets for more evaluation.

