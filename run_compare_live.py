"""
Live side-by-side training dashboard for A2C vs PPO.

Trains A2C then PPO sequentially in a single window:
  ┌──────────────────────┬──────────────────────┐
  │  A2C  Reward         │  PPO  Reward         │
  ├──────────────────────┼──────────────────────┤
  │  A2C  Entropy        │  PPO  Entropy        │
  └──────────────────────┴──────────────────────┘

A2C panels update live during A2C training.
When A2C finishes a "Done" stamp appears and PPO panels start updating.

Usage:
  python run_compare_live.py
  python run_compare_live.py --updates 500 --timesteps 500000   # quick test
  python run_compare_live.py --no-wandb
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import numpy as np
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from grid_env import AppleGridEnv
from train_teacher     import train as train_a2c
from train_teacher_ppo import train as train_ppo
from train_student_logit_distill import train_student
from compare           import random_baseline, oracle_baseline

# ── palette ───────────────────────────────────────────────────────────────────
BG_DARK   = "#0d1117"
BG_PANEL  = "#161b22"
C_BORDER  = "#30363d"
C_TICK    = "#8b949e"
C_GRID    = "#21262d"
C_A2C     = "#58a6ff"   # blue
C_PPO     = "#f85149"   # red
C_RAND    = "#ffa657"   # orange
C_ORACLE  = "#3fb950"   # green
C_DONE    = "#bc8cff"   # purple


def style_ax(ax):
    ax.set_facecolor(BG_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(C_BORDER)
    ax.tick_params(colors=C_TICK, labelsize=8)
    ax.yaxis.grid(True, alpha=0.25, color=C_GRID)
    ax.xaxis.grid(True, alpha=0.12, color=C_GRID)


def add_baselines(ax, rand_rew, oracle_rew):
    ax.axhline(rand_rew,   color=C_RAND,   lw=1.2, ls="--", alpha=0.8,
               label=f"Random  {rand_rew:.1f}")
    ax.axhline(oracle_rew, color=C_ORACLE, lw=1.2, ls="--", alpha=0.8,
               label=f"Oracle  {oracle_rew:.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--updates",    type=int,   default=4000,
                   help="A2C gradient steps")
    p.add_argument("--timesteps",  type=int,   default=2_500_000,
                   help="PPO total env steps")
    p.add_argument("--envs",       type=int,   default=16)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--no-wandb",   action="store_true")
    p.add_argument("--no-distil",  action="store_true",
                   help="Skip student distillation after training")
    args = p.parse_args()

    # ── baselines ─────────────────────────────────────────────────────────────
    print("Computing baselines…", flush=True)
    env = AppleGridEnv()
    RAND,   _ = random_baseline(env, 200)
    ORACLE, _ = oracle_baseline(env, 200)
    print(f"  Random: {RAND:.2f}   Oracle: {ORACLE:.2f}\n")

    # ── build figure ──────────────────────────────────────────────────────────
    plt.ion()
    fig = plt.figure(figsize=(14, 7), facecolor=BG_DARK)
    fig.suptitle("A2C  vs  PPO  —  Live Training Dashboard",
                 color="#58a6ff", fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.50, wspace=0.35,
                           left=0.07, right=0.97, top=0.91, bottom=0.10)

    ax_a2c_rew = fig.add_subplot(gs[0, 0])
    ax_a2c_ent = fig.add_subplot(gs[1, 0])
    ax_ppo_rew = fig.add_subplot(gs[0, 1])
    ax_ppo_ent = fig.add_subplot(gs[1, 1])

    for ax in [ax_a2c_rew, ax_a2c_ent, ax_ppo_rew, ax_ppo_ent]:
        style_ax(ax)

    # titles
    ax_a2c_rew.set_title("A2C  —  Apple Reward  (rolling 500 ep)",
                          color=C_A2C, fontsize=10, fontweight="bold")
    ax_a2c_ent.set_title("A2C  —  Policy Entropy",
                          color=C_A2C, fontsize=10, fontweight="bold")
    ax_ppo_rew.set_title("PPO  —  Apple Reward  (rolling 500 ep)",
                          color=C_PPO, fontsize=10, fontweight="bold")
    ax_ppo_ent.set_title("PPO  —  Policy Entropy",
                          color=C_PPO, fontsize=10, fontweight="bold")

    for ax in [ax_a2c_rew, ax_ppo_rew]:
        add_baselines(ax, RAND, ORACLE)
        ax.set_ylabel("Apples / ep", color=C_TICK, fontsize=8)
        ax.set_ylim(0, ORACLE * 1.08)
        ax.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=C_BORDER, labelcolor="white")

    for ax in [ax_a2c_ent, ax_ppo_ent]:
        ax.axhline(np.log(4), color=C_RAND, lw=1, ls="--", alpha=0.8,
                   label=f"Uniform  {np.log(4):.2f}")
        ax.set_ylabel("Entropy (nats)", color=C_TICK, fontsize=8)
        ax.set_ylim(0, np.log(4) * 1.15)
        ax.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=C_BORDER, labelcolor="white")

    for ax in [ax_a2c_ent, ax_ppo_ent]:
        ax.set_xlabel("Episodes", color=C_TICK, fontsize=8)

    # "Waiting…" placeholder text on PPO panels
    ppo_wait_rew = ax_ppo_rew.text(0.5, 0.5, "Waiting for A2C…",
                                    transform=ax_ppo_rew.transAxes,
                                    ha="center", va="center",
                                    color=C_TICK, fontsize=11, alpha=0.6)
    ppo_wait_ent = ax_ppo_ent.text(0.5, 0.5, "Waiting for A2C…",
                                    transform=ax_ppo_ent.transAxes,
                                    ha="center", va="center",
                                    color=C_TICK, fontsize=11, alpha=0.6)

    # live line objects
    a2c_rew_line, = ax_a2c_rew.plot([], [], color=C_A2C, lw=2)
    a2c_ent_line, = ax_a2c_ent.plot([], [], color=C_A2C, lw=2)
    ppo_rew_line, = ax_ppo_rew.plot([], [], color=C_PPO, lw=2)
    ppo_ent_line, = ax_ppo_ent.plot([], [], color=C_PPO, lw=2)

    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()

    # ── A2C callback ──────────────────────────────────────────────────────────
    a2c_eps, a2c_rews, a2c_ents = [], [], []

    def a2c_cb(ep, apples, entropy, critic_loss, shaping_coef):
        a2c_eps.append(ep);  a2c_rews.append(apples); a2c_ents.append(entropy)
        a2c_rew_line.set_data(a2c_eps, a2c_rews)
        a2c_ent_line.set_data(a2c_eps, a2c_ents)
        for ax in [ax_a2c_rew, ax_a2c_ent]:
            ax.relim(); ax.autoscale_view(scalex=True, scaley=False)
        try:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        except Exception:
            pass

    # ── PPO callback ──────────────────────────────────────────────────────────
    ppo_eps, ppo_rews, ppo_ents = [], [], []

    def ppo_cb(ep, apples, entropy, value_loss, clip_frac):
        ppo_eps.append(ep);  ppo_rews.append(apples); ppo_ents.append(entropy)
        ppo_rew_line.set_data(ppo_eps, ppo_rews)
        ppo_ent_line.set_data(ppo_eps, ppo_ents)
        for ax in [ax_ppo_rew, ax_ppo_ent]:
            ax.relim(); ax.autoscale_view(scalex=True, scaley=False)
        try:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        except Exception:
            pass

    # ── Step 1: train A2C ─────────────────────────────────────────────────────
    print("╔══════════════════════════════════════════════╗")
    print("║  Step 1 / 2  —  Training A2C teacher         ║")
    print("╚══════════════════════════════════════════════╝")
    train_a2c(
        n_updates   = args.updates,
        n_envs      = args.envs,
        lr          = args.lr,
        use_wandb   = not args.no_wandb,
        live_plot   = False,
        output      = "teacher_a2c.pt",
        log_callback = a2c_cb,
    )

    # stamp A2C done
    final_a2c = a2c_rews[-1] if a2c_rews else 0.0
    ax_a2c_rew.set_title(
        f"A2C  ✓  Done  —  {final_a2c:.2f} apples/ep",
        color=C_DONE, fontsize=10, fontweight="bold",
    )
    ppo_wait_rew.remove()
    ppo_wait_ent.remove()
    fig.canvas.draw()
    fig.canvas.flush_events()

    # ── Step 2: train PPO ─────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════╗")
    print("║  Step 2 / 2  —  Training PPO teacher         ║")
    print("╚══════════════════════════════════════════════╝")
    train_ppo(
        total_timesteps = args.timesteps,
        n_envs          = args.envs,
        lr              = args.lr,
        use_wandb       = not args.no_wandb,
        live_plot       = False,
        output          = "teacher_ppo.pt",
        log_callback    = ppo_cb,
    )

    # stamp PPO done
    final_ppo = ppo_rews[-1] if ppo_rews else 0.0
    ax_ppo_rew.set_title(
        f"PPO  ✓  Done  —  {final_ppo:.2f} apples/ep",
        color=C_DONE, fontsize=10, fontweight="bold",
    )
    fig.canvas.draw()
    fig.canvas.flush_events()

    # ── Step 3: distil ────────────────────────────────────────────────────────
    if not args.no_distil:
        print("\n--- Distilling A2C student ---")
        train_student(teacher_path="teacher_a2c.pt", output_path="student_a2c.pt")
        print("\n--- Distilling PPO student ---")
        train_student(teacher_path="teacher_ppo.pt", output_path="student_ppo.pt")

    print("\nAll done.  Close the window to exit.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
