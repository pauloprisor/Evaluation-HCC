#!/bin/bash

# exit on error
set -e

echo "--- Initializing project environment ---"

# venv setup
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Created venv"
fi

source venv/bin/activate

# install core deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install the rest from requirements
pip install -r requirements.txt

# needed for sentence splitting in our method
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# create necessary folders if they don't exist
mkdir -p results
mkdir -p data

echo "--- Downloading Full LongBench Dataset ---"
python -c "
from datasets import load_dataset
import os, json

tasks = [
    'narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', 
    '2wikimqa', 'musique', 'gov_report', 'qmsum', 'multi_news', 
    'trec', 'triviaqa', 'samsum', 'passage_count', 
    'passage_retrieval_en', 'lcc', 'repobench-p'
]
os.makedirs('data', exist_ok=True)

for task in tasks:
    if os.path.exists(f'data/{task}.jsonl'):
        print(f'Skipping {task} (already exists)')
        continue
    print(f'Downloading {task}...')
    try:
        ds = load_dataset('THUDM/LongBench', task, split='test', trust_remote_code=True)
        with open(f'data/{task}.jsonl', 'w', encoding='utf-8') as f:
            for entry in ds:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f'Error downloading {task}: {e}')
"


echo "Done. Use 'source venv/bin/activate' to start."
