#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update

sudo apt-get install python3.10 python3.10-venv -y

VENV_DIR="./venv"

python3.10 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

if [ -f ./requirements.txt ]; then
    pip install -r ./requirements.txt
else
    echo "requirements.txt is not found. Skipping dependencies installing."
fi

echo "venv is configured successfully."
