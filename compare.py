"""
Evaluate and compare all trained models side-by-side.

Loads any combination of teacher_a2c.pt, teacher_ppo.pt,
student_a2c.pt, student_ppo.pt (gracefully skips missing files),
runs N stochastic episodes per model, and prints a formatted table.

Usage:
  python compare.py                    # defaults: all standard filenames, 200 ep
  python compare.py --episodes 500
  python compare.py --teacher-a2c teacher.pt --student-a2c student.pt
"""

import argparse
import os

import numpy as np
import torch
from torch.distributions import Categorical

from grid_env import AppleGridEnv
from models import TeacherNet, StudentNet


# ── device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── baselines ─────────────────────────────────────────────────────────────────
def random_baseline(env, n_ep):
    results = []
    for _ in range(n_ep):
        env.reset()
        done = False
        ep = 0.0
        while not done:
            _, _, done, info = env.step(np.random.randint(4))
            ep += info.get("apple_reward", 0)
        results.append(ep)
    return float(np.mean(results)), float(np.std(results))


def oracle_baseline(env, n_ep):
    results = []
    for _ in range(n_ep):
        env.reset()
        done = False
        ep = 0.0
        while not done:
            apples = np.argwhere(env.grid == 1.0)
            if len(apples) == 0:
                action = np.random.randint(4)
            else:
                dists   = np.abs(apples[:, 0] - env.row) + np.abs(apples[:, 1] - env.col)
                nearest = apples[np.argmin(dists)]
                dr, dc  = nearest[0] - env.row, nearest[1] - env.col
                action  = (0 if dr < 0 else 1) if abs(dr) >= abs(dc) else (2 if dc < 0 else 3)
            _, _, done, info = env.step(action)
            ep += info.get("apple_reward", 0)
        results.append(ep)
    return float(np.mean(results)), float(np.std(results))


# ── model evaluation ──────────────────────────────────────────────────────────
@torch.no_grad()
def eval_model(model, env, device, n_ep):
    """Stochastic evaluation; returns (mean, std) apple reward."""
    results = []
    for _ in range(n_ep):
        obs, done = env.reset(), False
        ep = 0.0
        while not done:
            x      = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            out    = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            action = Categorical(logits=logits).sample().item()
            obs, _, done, info = env.step(action)
            ep += info.get("apple_reward", 0)
        results.append(ep)
    return float(np.mean(results)), float(np.std(results))


def load_teacher(path, env, device):
    m = TeacherNet(env.obs_size, env.n_actions).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    return m


def load_student(path, env, device):
    m = StudentNet(env.obs_size, env.n_actions).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    return m


# ── formatting helpers ────────────────────────────────────────────────────────
def param_count(model):
    return sum(p.numel() for p in model.parameters())


def pct_of_gap(mean, rand_mean, oracle_mean):
    gap = oracle_mean - rand_mean
    if gap < 1e-6:
        return 0.0
    return 100 * (mean - rand_mean) / gap


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Compare all trained models")
    p.add_argument("--teacher-a2c",  type=str, default="teacher_a2c.pt")
    p.add_argument("--teacher-ppo",  type=str, default="teacher_ppo.pt")
    p.add_argument("--student-a2c",  type=str, default="student_a2c.pt")
    p.add_argument("--student-ppo",  type=str, default="student_ppo.pt")
    p.add_argument("--episodes",     type=int, default=200,
                   help="Evaluation episodes per model")
    args = p.parse_args()

    device = get_device()
    env    = AppleGridEnv()
    n_ep   = args.episodes

    print(f"\nDevice: {device}   Episodes per model: {n_ep}")
    print("Computing baselines…", flush=True)

    rand_mean, rand_std     = random_baseline(env, n_ep)
    oracle_mean, oracle_std = oracle_baseline(env, n_ep)

    # ── collect rows ──────────────────────────────────────────────────────────
    rows = []   # (label, params_str, mean, std)
    rows.append(("Random",          "—",         rand_mean,   rand_std))
    rows.append(("Oracle (greedy)", "—",         oracle_mean, oracle_std))

    model_specs = [
        (args.teacher_a2c, "Teacher A2C", load_teacher),
        (args.teacher_ppo, "Teacher PPO", load_teacher),
        (args.student_a2c, "Student A2C", load_student),
        (args.student_ppo, "Student PPO", load_student),
    ]

    for path, label, loader in model_specs:
        if not os.path.exists(path):
            print(f"  [{label}] not found at {path!r} — skipping")
            continue
        print(f"  Evaluating {label} ({path})…", flush=True)
        model = loader(path, env, device)
        n_p   = param_count(model)
        mean, std = eval_model(model, env, device, n_ep)
        rows.append((label, f"{n_p:,}", mean, std))

    # ── print table ───────────────────────────────────────────────────────────
    col_w = [18, 9, 9, 5, 9]
    sep   = "  ".join("─" * w for w in col_w)
    hdr   = (f"{'Model':<{col_w[0]}}  {'Params':>{col_w[1]}}  "
             f"{'Apples/ep':>{col_w[2]}}  {'±std':>{col_w[3]}}  "
             f"{'vs oracle':>{col_w[4]}}")

    print(f"\n{'━'*60}")
    print(f"  Comparison  ({n_ep} stochastic episodes each)")
    print(f"{'━'*60}")
    print(f"  {hdr}")
    print(f"  {sep}")

    for label, params, mean, std in rows:
        pct = pct_of_gap(mean, rand_mean, oracle_mean)
        print(
            f"  {label:<{col_w[0]}}  {params:>{col_w[1]}}  "
            f"{mean:>{col_w[2]}.2f}  {std:>{col_w[3]}.1f}  "
            f"{pct:>{col_w[4]-1}.0f}%"
        )

    print(f"{'━'*60}\n")


if __name__ == "__main__":
    main()
