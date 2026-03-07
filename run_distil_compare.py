"""
Live side-by-side distillation comparison dashboard.

Trains both distillation methods sequentially using a pre-trained teacher,
showing how quickly each student learns in real time:

  ┌──────────────────────┬──────────────────────┐
  │  Logit Distill       │  RL Distill          │
  │  (Forward KL)        │  (Reverse KL via PPO)│
  │  Student reward      │  Student reward      │
  ├──────────────────────┼──────────────────────┤
  │  KL loss (↓ = good)  │  KL bonus (↑ = good) │
  └──────────────────────┴──────────────────────┘

X-axis for both: approximate env steps consumed.

The forward-KL method (logit distill) trains on teacher rollouts —
steps are teacher environment interactions.  The reverse-KL method
(RL distill) trains on student rollouts — steps are student interactions.
Both are directly comparable as "steps paid to produce each student update".

Usage:
  python run_distil_compare.py --teacher teacher_ppo.pt
  python run_distil_compare.py --teacher teacher_a2c.pt --logit-iters 100 --rl-updates 250
  python run_distil_compare.py --no-save   # skip saving weights
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from grid_env import AppleGridEnv
from models   import TeacherNet
from compare  import random_baseline, oracle_baseline
from train_student_logit_distill import train_student, evaluate_stochastic, get_device
from train_student_rl_distill    import train_student_rl

# ── palette ───────────────────────────────────────────────────────────────────
BG_DARK  = "#0d1117"
BG_PANEL = "#161b22"
C_BORDER = "#30363d"
C_TICK   = "#8b949e"
C_GRID   = "#21262d"
C_LOGIT  = "#58a6ff"   # blue — forward KL
C_RL     = "#f85149"   # red  — reverse KL
C_RAND   = "#ffa657"   # orange
C_ORACLE = "#3fb950"   # green
C_DONE   = "#bc8cff"   # purple


def style_ax(ax):
    ax.set_facecolor(BG_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(C_BORDER)
    ax.tick_params(colors=C_TICK, labelsize=8)
    ax.yaxis.grid(True, alpha=0.25, color=C_GRID)
    ax.xaxis.grid(True, alpha=0.12, color=C_GRID)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher",      type=str,   default="teacher_ppo.pt",
                   help="Pre-trained teacher weights to distil from")
    p.add_argument("--logit-iters",  type=int,   default=200,
                   help="Iterations for logit (forward-KL) distillation")
    p.add_argument("--rl-updates",   type=int,   default=100,
                   help="PPO updates for RL (reverse-KL) distillation"
                        " (100 updates × 128 steps × 16 envs ≈ 200K steps,"
                        " comparable to logit's 200 iters × 640 steps = 128K)")
    p.add_argument("--envs",         type=int,   default=16)
    p.add_argument("--kl-coef",      type=float, default=0.02,
                   help="KL bonus weight for RL distillation")
    p.add_argument("--no-save",      action="store_true",
                   help="Do not save student weights")
    args = p.parse_args()

    # ── baselines + teacher eval ──────────────────────────────────────────────
    print("Computing baselines and teacher performance…", flush=True)
    env    = AppleGridEnv()
    device = get_device()
    RAND,   _ = random_baseline(env, 200)
    ORACLE, _ = oracle_baseline(env, 200)

    teacher_model = TeacherNet(env.obs_size, env.n_actions).to(device)
    teacher_model.load_state_dict(
        torch.load(args.teacher, map_location=device, weights_only=True)
    )
    teacher_model.eval()
    TEACHER = evaluate_stochastic(teacher_model, env, device, n_episodes=50)
    print(f"  Random: {RAND:.2f}   Teacher: {TEACHER:.2f}   Oracle: {ORACLE:.2f}\n")

    # ── figure ────────────────────────────────────────────────────────────────
    plt.ion()
    fig = plt.figure(figsize=(14, 7), facecolor=BG_DARK)
    fig.suptitle(
        "Distillation Comparison  —  Forward KL (logit)  vs  Reverse KL (RL)",
        color="#58a6ff", fontsize=13, fontweight="bold",
    )

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.50, wspace=0.35,
                           left=0.07, right=0.97, top=0.91, bottom=0.10)

    ax_l_rew = fig.add_subplot(gs[0, 0])
    ax_l_kl  = fig.add_subplot(gs[1, 0])
    ax_r_rew = fig.add_subplot(gs[0, 1])
    ax_r_kl  = fig.add_subplot(gs[1, 1])

    for ax in [ax_l_rew, ax_l_kl, ax_r_rew, ax_r_kl]:
        style_ax(ax)

    ax_l_rew.set_title("Logit Distill (Forward KL)  —  Student Reward",
                        color=C_LOGIT, fontsize=10, fontweight="bold")
    ax_l_kl.set_title( "Logit Distill  —  KL Loss  (lower = better)",
                        color=C_LOGIT, fontsize=10, fontweight="bold")
    ax_r_rew.set_title("RL Distill (Reverse KL)  —  Student Reward",
                        color=C_RL, fontsize=10, fontweight="bold")
    ax_r_kl.set_title( "RL Distill  —  KL Bonus  (higher = teacher agrees)",
                        color=C_RL, fontsize=10, fontweight="bold")

    C_TEACHER = "#e3b341"   # gold — teacher reference
    for ax in [ax_l_rew, ax_r_rew]:
        ax.axhline(RAND,    color=C_RAND,    lw=1.2, ls="--", alpha=0.8,
                   label=f"Random   {RAND:.1f}")
        ax.axhline(TEACHER, color=C_TEACHER, lw=1.5, ls="-.",  alpha=0.9,
                   label=f"Teacher  {TEACHER:.1f}")
        ax.axhline(ORACLE,  color=C_ORACLE,  lw=1.2, ls="--", alpha=0.8,
                   label=f"Oracle   {ORACLE:.1f}")
        ax.set_ylabel("Apples / ep", color=C_TICK, fontsize=8)
        ax.set_ylim(0, ORACLE * 1.08)
        ax.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=C_BORDER, labelcolor="white")

    ax_l_kl.set_ylabel("KL loss",    color=C_TICK, fontsize=8)
    ax_r_kl.set_ylabel("KL bonus",   color=C_TICK, fontsize=8)
    for ax in [ax_l_kl, ax_r_kl]:
        ax.set_xlabel("Env steps", color=C_TICK, fontsize=8)
    for ax in [ax_l_rew, ax_r_rew]:
        ax.set_xlabel("Env steps", color=C_TICK, fontsize=8)

    # "Waiting…" placeholder on RL panels
    rl_wait_rew = ax_r_rew.text(0.5, 0.5, "Waiting for logit distill…",
                                 transform=ax_r_rew.transAxes,
                                 ha="center", va="center",
                                 color=C_TICK, fontsize=11, alpha=0.6)
    rl_wait_kl  = ax_r_kl.text( 0.5, 0.5, "Waiting for logit distill…",
                                 transform=ax_r_kl.transAxes,
                                 ha="center", va="center",
                                 color=C_TICK, fontsize=11, alpha=0.6)

    # live lines
    l_rew_line, = ax_l_rew.plot([], [], color=C_LOGIT, lw=2)
    l_kl_line,  = ax_l_kl.plot( [], [], color=C_LOGIT, lw=1.5)
    r_rew_line, = ax_r_rew.plot([], [], color=C_RL,    lw=2)
    r_kl_line,  = ax_r_kl.plot( [], [], color=C_RL,    lw=1.5)

    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()

    # ── logit distill callback ────────────────────────────────────────────────
    l_steps, l_rews, l_kls = [], [], []

    def logit_cb(env_steps, kl_loss, teacher_rew, student_rew):
        l_steps.append(env_steps)
        l_rews.append(student_rew)
        l_kls.append(kl_loss)
        l_rew_line.set_data(l_steps, l_rews)
        l_kl_line.set_data( l_steps, l_kls)
        for ax in [ax_l_rew, ax_l_kl]:
            ax.relim(); ax.autoscale_view(scalex=True, scaley=False)
        try:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        except Exception:
            pass

    # ── RL distill callback ───────────────────────────────────────────────────
    r_steps, r_rews, r_kls = [], [], []

    def rl_cb(env_steps, student_rew, avg_kl_bonus, entropy):
        r_steps.append(env_steps)
        r_rews.append(student_rew)
        r_kls.append(avg_kl_bonus)
        r_rew_line.set_data(r_steps, r_rews)
        r_kl_line.set_data( r_steps, r_kls)
        for ax in [ax_r_rew, ax_r_kl]:
            ax.relim(); ax.autoscale_view(scalex=True, scaley=False)
        try:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        except Exception:
            pass

    # ── Step 1: logit distillation ────────────────────────────────────────────
    print("╔══════════════════════════════════════════════╗")
    print("║  Step 1 / 2  —  Logit distillation (fwd KL) ║")
    print("╚══════════════════════════════════════════════╝")
    logit_out = "student_logit.pt" if not args.no_save else "/dev/null"
    train_student(
        n_iterations = args.logit_iters,
        n_envs       = args.envs,
        teacher_path = args.teacher,
        output_path  = logit_out,
        log_callback = logit_cb,
    )

    final_logit = l_rews[-1] if l_rews else 0.0
    ax_l_rew.set_title(
        f"Logit Distill  ✓  Done  —  {final_logit:.2f} apples/ep",
        color=C_DONE, fontsize=10, fontweight="bold",
    )
    rl_wait_rew.remove()
    rl_wait_kl.remove()
    fig.canvas.draw()
    fig.canvas.flush_events()

    # ── Step 2: RL distillation ───────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════╗")
    print("║  Step 2 / 2  —  RL distillation (rev KL)    ║")
    print("╚══════════════════════════════════════════════╝")
    rl_out = "student_rl.pt" if not args.no_save else "/dev/null"
    train_student_rl(
        n_updates    = args.rl_updates,
        n_envs       = args.envs,
        kl_coef      = args.kl_coef,
        teacher_path = args.teacher,
        output_path  = rl_out,
        log_callback = rl_cb,
    )

    final_rl = r_rews[-1] if r_rews else 0.0
    ax_r_rew.set_title(
        f"RL Distill  ✓  Done  —  {final_rl:.2f} apples/ep",
        color=C_DONE, fontsize=10, fontweight="bold",
    )
    fig.canvas.draw()
    fig.canvas.flush_events()

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "━" * 55)
    print("  Distillation Comparison Summary")
    print("━" * 55)
    gap = ORACLE - RAND
    print(f"  Teacher  ({args.teacher}) : {TEACHER:.2f}  ({100*(TEACHER-RAND)/gap:.0f}% of oracle gap)")
    print(f"  Random baseline          : {RAND:.2f}")
    print(f"  Oracle baseline          : {ORACLE:.2f}")
    print(f"  Logit distill (fwd KL)   : {final_logit:.2f}  ({100*(final_logit-RAND)/gap:.0f}% of oracle gap)")
    print(f"  RL distill    (rev KL)   : {final_rl:.2f}  ({100*(final_rl-RAND)/gap:.0f}% of oracle gap)")
    print("━" * 55)

    print("\nAll done.  Close the window to exit.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
