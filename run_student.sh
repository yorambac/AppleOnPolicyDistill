#!/usr/bin/env bash
# Distil a smaller student from teacher.pt and save student.pt
#
# Usage:
#   bash run_student.sh
#
# Requires teacher.pt to exist (run run_teacher.sh first).

set -euo pipefail

PYTHON="$(conda run -n try which python)"

# Work around duplicate OpenMP library on macOS + MKL
export KMP_DUPLICATE_LIB_OK=TRUE

cd "$(dirname "$0")"

if [ ! -f teacher.pt ]; then
    echo "Error: teacher.pt not found. Run 'bash run_teacher.sh' first." >&2
    exit 1
fi

echo "=== Student distillation ==="
"$PYTHON" train_student.py "$@"
