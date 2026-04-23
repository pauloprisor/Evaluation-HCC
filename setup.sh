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

echo "Done. Use 'source venv/bin/activate' to start."
