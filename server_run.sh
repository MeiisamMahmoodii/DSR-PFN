#!/bin/bash
# Server Training Script for ISD-CP
# Usage: bash server_run.sh

# 1. Update Code
echo "--- Pulling Latest Code ---"
git pull origin main

# 2. Setup Venv (if needed)
if [ ! -d ".venv" ]; then
    echo "--- Creating Virtual Env ---"
    python3 -m venv .venv
fi
source .venv/bin/activate

# 3. Install Dependencies
echo "--- Installing Dependencies ---"
pip install -r requirements.txt

# 4. Run Sparsity Finetuning (High-SHD Fix)
# 4. Run Training
echo "--- Starting Training ---"
# Checkpoint logic handled by main.py if passed args
python main.py "$@"

echo "--- Training Complete ---"
