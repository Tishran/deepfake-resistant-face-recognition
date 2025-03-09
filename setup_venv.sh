#!/bin/bash

# Install required Python virtual environment package
sudo apt-get update
sudo apt-get install python3.10-venv -y

# Create virtual environment
VENV_DIR="./venv"
python3 -m venv "$VENV_DIR"

# Activate the virtual environment and install dependencies
source "$VENV_DIR/bin/activate"
pip install -r ./working/razrabotat-model-raspoznavaniya-lic-ustojchivoj-k-atakam-s-ispolzovaniem-deepfake-6468/requirements.txt

echo "Virtual environment setup complete."