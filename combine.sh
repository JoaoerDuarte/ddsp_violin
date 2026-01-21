#!/bin/bash

# This script finds all .py files in the current directory and its subdirectories,
# excluding specific folders, and combines them into a single file.

echo "🔍 Finding and combining Python files..."

find . -type f -name "*.py" \
-not -path "*/.ipynb_checkpoints/*" \
-not -path "./.venv/*" \
-not -path "*/__pycache__/*" \
-exec sh -c 'echo "\n--- {} ---\n" && cat "{}"' \; > repo.txt

echo "Done! Project combined into repo.txt"
