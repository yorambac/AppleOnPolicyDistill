#!/usr/bin/env bash
# Train the A2C teacher and save teacher.pt
#
# Usage:
#   bash run_teacher.sh                  # defaults: 4000 updates, 16 envs
#   bash run_teacher.sh --updates 8000   # longer run
#   bash run_teacher.sh --no-plot        # headless (no matplotlib window)
#   bash run_teacher.sh --no-wandb       # disable W&B (enabled by default if wandb is installed)

set -euo pipefail

PYTHON="$(conda run -n try which python)"

# Work around duplicate OpenMP library on macOS + MKL
export KMP_DUPLICATE_LIB_OK=TRUE

cd "$(dirname "$0")"

echo "=== Teacher training ==="
"$PYTHON" train_teacher.py "$@"
