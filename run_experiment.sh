#!/usr/bin/env bash
# Full experiment: train teacher → distil student → visualise student
#
# Usage:
#   bash run_experiment.sh                        # all defaults
#   bash run_experiment.sh --no-plot              # headless training, viz at end
#   bash run_experiment.sh --updates 2000 --no-wandb  # quick smoke-test
#
# Teacher flags forwarded verbatim: --updates, --envs, --lr, --no-wandb, --no-plot
# Visualisation always opens 3 stochastic episodes after distillation.

set -euo pipefail

PYTHON="$(conda run -n try which python)"
export KMP_DUPLICATE_LIB_OK=TRUE

cd "$(dirname "$0")"

# ── Step 1: teacher ────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Step 1 / 3  —  Teacher training (A2C)       ║"
echo "╚══════════════════════════════════════════════╝"
"$PYTHON" train_teacher.py "$@"

# ── Step 2: student ────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Step 2 / 3  —  Student distillation (KL)    ║"
echo "╚══════════════════════════════════════════════╝"
if [ ! -f teacher.pt ]; then
    echo "Error: teacher.pt not found after training step." >&2
    exit 1
fi
"$PYTHON" train_student.py

# ── Step 3: visualise ──────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Step 3 / 3  —  Visualise student policy     ║"
echo "╚══════════════════════════════════════════════╝"
if [ ! -f student.pt ]; then
    echo "Error: student.pt not found after distillation step." >&2
    exit 1
fi
"$PYTHON" visualize_student.py --n 3

echo ""
echo "Done.  Weights saved: teacher.pt  student.pt"
