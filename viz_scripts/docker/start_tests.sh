#!/bin/bash
set -e  # Exit on error

# change python environment
pwd
source setup/activate.sh || exit 1
conda env list
cd saved-notebooks/tests || exit 1

echo "Starting unit tests..."
PYTHONPATH=../.. coverage run -m pytest . -v

coverage report
