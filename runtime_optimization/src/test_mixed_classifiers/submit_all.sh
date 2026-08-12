#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for script in "$SCRIPT_DIR"/test_classifiers_on_embeddings_*.sh; do
    sbatch "$script"
    echo "Submitted: $script"
done
