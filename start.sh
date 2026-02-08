#!/usr/bin/env bash
set -e

# Go to project root (directory of this script)
cd "$(dirname "$0")"

# Initialize pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"

eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Activate the project env (from .python-version)
pyenv activate emotion-env

# Sanity check
echo "Using Python: $(python --version)"
echo "Python path: $(which python)"

# Run the app
python main.py
