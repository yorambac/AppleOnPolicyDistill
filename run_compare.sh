#!/usr/bin/env bash
# Full comparison experiment:
#   Step 1  Train A2C teacher  → teacher_a2c.pt
#   Step 2  Train PPO teacher  → teacher_ppo.pt
#   Step 3  Distil both        → student_a2c.pt, student_ppo.pt
#   Step 4  Compare all four   → printed table
#
# Usage:
#   bash run_compare.sh                         # full run (~35 min)
#   bash run_compare.sh --no-plot --no-wandb    # headless
#   bash run_compare.sh --no-plot --updates 500 --timesteps 500000  # quick smoke-test
#
# Flags forwarded to BOTH teacher scripts:
#   --no-plot   disable live matplotlib windows
#   --no-wandb  disable Weights & Biases logging
#
# A2C-only flags (via --updates, --envs, --lr):
#   --updates N   number of A2C gradient steps   (default 4000)
#   --envs N      parallel environments           (default 16)
#   --lr LR       learning rate                   (default 3e-4)
#
# PPO-only flags (via --timesteps, --steps, --epochs, --clip):
#   --timesteps N  total env steps                (default 2500000)
#   --steps N      rollout length per env         (default 128)
#   --epochs N     PPO re-use epochs              (default 4)
#   --clip F       clip epsilon                   (default 0.2)

set -euo pipefail

PYTHON="$(conda run -n try which python)"
export KMP_DUPLICATE_LIB_OK=TRUE

cd "$(dirname "$0")"

# ── parse flags ────────────────────────────────────────────────────────────────
# Flags that apply to both algorithms go into COMMON_FLAGS.
# Algorithm-specific flags go into A2C_FLAGS / PPO_FLAGS.
COMMON_FLAGS=()
A2C_FLAGS=()
PPO_FLAGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-plot|--no-wandb)
            COMMON_FLAGS+=("$1"); shift ;;
        --updates|--envs|--lr)
            A2C_FLAGS+=("$1" "$2"); shift 2 ;;
        --timesteps|--steps|--epochs|--clip)
            PPO_FLAGS+=("$1" "$2"); shift 2 ;;
        *)
            echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# ── Step 1: A2C teacher ────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Step 1 / 4  —  A2C teacher training             ║"
echo "╚══════════════════════════════════════════════════╝"
"$PYTHON" train_teacher.py \
    "${COMMON_FLAGS[@]+"${COMMON_FLAGS[@]}"}" \
    "${A2C_FLAGS[@]+"${A2C_FLAGS[@]}"}" \
    --output teacher_a2c.pt

if [ ! -f teacher_a2c.pt ]; then
    echo "Error: teacher_a2c.pt not produced." >&2; exit 1
fi

# ── Step 2: PPO teacher ────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Step 2 / 4  —  PPO teacher training             ║"
echo "╚══════════════════════════════════════════════════╝"
"$PYTHON" train_teacher_ppo.py \
    "${COMMON_FLAGS[@]+"${COMMON_FLAGS[@]}"}" \
    "${PPO_FLAGS[@]+"${PPO_FLAGS[@]}"}" \
    --output teacher_ppo.pt

if [ ! -f teacher_ppo.pt ]; then
    echo "Error: teacher_ppo.pt not produced." >&2; exit 1
fi

# ── Step 3: distil both students ──────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Step 3 / 4  —  Student distillation             ║"
echo "╚══════════════════════════════════════════════════╝"

echo "--- Distilling A2C student ---"
"$PYTHON" train_student.py --teacher teacher_a2c.pt --output student_a2c.pt

if [ ! -f student_a2c.pt ]; then
    echo "Error: student_a2c.pt not produced." >&2; exit 1
fi

echo ""
echo "--- Distilling PPO student ---"
"$PYTHON" train_student.py --teacher teacher_ppo.pt --output student_ppo.pt

if [ ! -f student_ppo.pt ]; then
    echo "Error: student_ppo.pt not produced." >&2; exit 1
fi

# ── Step 4: compare ────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Step 4 / 4  —  Final comparison                 ║"
echo "╚══════════════════════════════════════════════════╝"
"$PYTHON" compare.py \
    --teacher-a2c teacher_a2c.pt \
    --teacher-ppo teacher_ppo.pt \
    --student-a2c student_a2c.pt \
    --student-ppo student_ppo.pt

echo "All weights saved:"
echo "  teacher_a2c.pt   teacher_ppo.pt"
echo "  student_a2c.pt   student_ppo.pt"
