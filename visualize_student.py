"""
Visualize a student policy episode in real time.

Layout
──────
┌─────────────────────┬──────────────────────┐
│                     │  Action probs (bars) │
│   15×15 grid        ├──────────────────────┤
│   @=agent  A=apple  │  Cumulative reward   │
│                     │  curve               │
└─────────────────────┴──────────────────────┘

Run:
  python visualize_student.py              # 3 episodes, auto-advance
  python visualize_student.py --n 5       # 5 episodes
  python visualize_student.py --delay 0.1 # faster
  python visualize_student.py --greedy    # argmax instead of sampling
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from grid_env import AppleGridEnv
from models import StudentNet

matplotlib.rcParams.update({
    "figure.facecolor": "#0d1117",
    "text.color":       "white",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "white",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
})

# ── colour palette ─────────────────────────────────────────────────────────────
BG_DARK   = "#0d1117"
BG_PANEL  = "#161b22"
C_EMPTY   = "#21262d"
C_APPLE   = "#ffa657"   # orange-yellow
C_AGENT   = "#3fb950"   # bright green
C_BORDER  = "#30363d"
C_ACCENT  = "#58a6ff"   # blue
C_RED     = "#f85149"

ACTION_LABELS  = ["↑  Up", "↓  Down", "←  Left", "→  Right"]
ACTION_COLORS  = ["#58a6ff", "#f85149", "#bc8cff", "#ffa657"]

# ── device ─────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── grid → RGB image ───────────────────────────────────────────────────────────
def grid_to_rgb(env: AppleGridEnv) -> np.ndarray:
    """Convert current env state to an (H, W, 3) float32 RGB array."""
    from matplotlib.colors import to_rgb
    ec = np.array(to_rgb(C_EMPTY))
    ac = np.array(to_rgb(C_APPLE))
    gc = np.array(to_rgb(C_AGENT))

    img = np.full((env.grid_size, env.grid_size, 3), ec)
    img[env.grid == 1.0] = ac
    img[env.row, env.col] = gc
    return img


# ── single episode visualisation ───────────────────────────────────────────────
def run_episode(model: StudentNet, env: AppleGridEnv, device, delay: float, greedy: bool):
    obs  = env.reset()
    done = False
    total_reward  = 0.0
    step          = 0
    reward_hist   = []   # per-step reward
    cum_hist      = []   # cumulative

    # ── build figure ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 7), facecolor=BG_DARK)
    gs  = gridspec.GridSpec(
        2, 2,
        figure=fig,
        width_ratios=[1.6, 1],
        hspace=0.45,
        wspace=0.35,
        left=0.05, right=0.97, top=0.90, bottom=0.08,
    )
    ax_grid  = fig.add_subplot(gs[:, 0])
    ax_probs = fig.add_subplot(gs[0, 1])
    ax_rew   = fig.add_subplot(gs[1, 1])

    sup = fig.suptitle(
        "Student Policy  —  On-Policy Distillation Demo",
        fontsize=13, fontweight="bold", color=C_ACCENT, y=0.97,
    )

    # ── grid axes setup ───────────────────────────────────────────────────────
    ax_grid.set_facecolor(BG_PANEL)
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    for sp in ax_grid.spines.values():
        sp.set_edgecolor(C_BORDER)

    im = ax_grid.imshow(
        grid_to_rgb(env), interpolation="nearest", aspect="equal",
    )
    # grid lines
    gs_ = env.grid_size
    for k in range(gs_ + 1):
        ax_grid.axhline(k - 0.5, color=C_BORDER, lw=0.6, zorder=2)
        ax_grid.axvline(k - 0.5, color=C_BORDER, lw=0.6, zorder=2)

    grid_title = ax_grid.set_title(
        f"Step 0/{env.episode_length}   Apples: 0   Policy: {'greedy' if greedy else 'stochastic'}",
        color="white", fontsize=10, pad=8,
    )

    # legend patches
    legend = ax_grid.legend(
        handles=[
            mpatches.Patch(color=C_AGENT, label="Agent (@)"),
            mpatches.Patch(color=C_APPLE, label="Apple (A)"),
        ],
        loc="upper right", fontsize=8,
        facecolor=BG_PANEL, edgecolor=C_BORDER, labelcolor="white",
        handlelength=1.2, framealpha=0.8,
    )

    # ── prob bar axes setup ────────────────────────────────────────────────────
    ax_probs.set_facecolor(BG_PANEL)
    for sp in ax_probs.spines.values():
        sp.set_edgecolor(C_BORDER)
    ax_probs.set_title("Action Probabilities", color="white", fontsize=10, pad=6)
    ax_probs.set_ylim(0, 1.05)
    ax_probs.set_xticks(range(4))
    ax_probs.set_xticklabels(["↑", "↓", "←", "→"], fontsize=13)
    ax_probs.yaxis.grid(True, alpha=0.3)

    bars     = ax_probs.bar(range(4), [0.25]*4, color=ACTION_COLORS,
                            edgecolor=C_BORDER, linewidth=1.5)
    prob_txt = [
        ax_probs.text(i, 0.28, "", ha="center", va="bottom",
                      color="white", fontsize=9, fontweight="bold")
        for i in range(4)
    ]
    chosen_txt = ax_probs.text(
        0.5, 0.93, "", ha="center", va="top",
        transform=ax_probs.transAxes, color=C_ACCENT,
        fontsize=10, fontweight="bold",
    )

    # ── reward axes setup ──────────────────────────────────────────────────────
    ax_rew.set_facecolor(BG_PANEL)
    for sp in ax_rew.spines.values():
        sp.set_edgecolor(C_BORDER)
    ax_rew.set_title("Cumulative Reward", color="white", fontsize=10, pad=6)
    ax_rew.set_xlim(0, env.episode_length)
    ax_rew.set_ylim(0, env.n_apples_init + 4)
    ax_rew.set_xlabel("Step", color="#8b949e", fontsize=9)
    ax_rew.yaxis.grid(True, alpha=0.3)
    rew_line, = ax_rew.plot([], [], color=C_RED, linewidth=2.5)
    rew_fill  = [None]   # list so we can mutate from inner scope
    rew_txt   = ax_rew.text(
        0.97, 0.88, "0", ha="right", va="top",
        transform=ax_rew.transAxes, color=C_RED,
        fontsize=14, fontweight="bold",
    )

    plt.ion()
    plt.show()

    # ── episode loop ──────────────────────────────────────────────────────────
    while not done:
        # forward pass
        x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs  = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        if greedy:
            action = int(probs.argmax())
        else:
            action = int(np.random.choice(4, p=probs))

        # ── update grid ───────────────────────────────────────────────────────
        im.set_data(grid_to_rgb(env))
        grid_title.set_text(
            f"Step {step:>2}/{env.episode_length}   "
            f"Apples: {int(total_reward):>2}   "
            f"Action: {ACTION_LABELS[action]}"
        )

        # ── update prob bars ──────────────────────────────────────────────────
        for i, (bar, p) in enumerate(zip(bars, probs)):
            bar.set_height(p)
            bar.set_linewidth(3.0 if i == action else 1.0)
            bar.set_edgecolor("white" if i == action else C_BORDER)
            prob_txt[i].set_text(f"{p:.2f}")
            prob_txt[i].set_y(p + 0.02)
        chosen_txt.set_text(ACTION_LABELS[action])

        # ── update reward curve ───────────────────────────────────────────────
        nonlocal_cum = np.cumsum(cum_hist) if cum_hist else np.array([0])
        xs = np.arange(len(nonlocal_cum))
        rew_line.set_data(xs, nonlocal_cum)
        if rew_fill[0] is not None:
            rew_fill[0].remove()
        rew_fill[0] = ax_rew.fill_between(xs, nonlocal_cum, alpha=0.25, color=C_RED)
        rew_txt.set_text(f"{total_reward:.0f}")

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(delay)

        # step
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        reward_hist.append(reward)
        cum_hist.append(total_reward)
        step += 1

    # ── final frame ───────────────────────────────────────────────────────────
    im.set_data(grid_to_rgb(env))
    grid_title.set_text(
        f"DONE   Total apples: {int(total_reward)}   "
        f"({'GREAT!' if total_reward >= env.n_apples_init * 0.6 else 'OK'})"
    )
    grid_title.set_color(C_ACCENT)

    cum_arr = np.cumsum(cum_hist)
    rew_line.set_data(np.arange(len(cum_arr)), cum_arr)
    if rew_fill[0] is not None:
        rew_fill[0].remove()
    ax_rew.fill_between(np.arange(len(cum_arr)), cum_arr, alpha=0.25, color=C_RED)
    rew_txt.set_text(f"{total_reward:.0f}")

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(1.5)
    plt.close(fig)

    return total_reward


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Visualize student policy")
    parser.add_argument("--n",      type=int,   default=3,    help="Number of episodes")
    parser.add_argument("--delay",  type=float, default=0.25, help="Seconds per step")
    parser.add_argument("--greedy", action="store_true",      help="Use greedy (argmax) policy")
    args = parser.parse_args()

    device = get_device()
    env    = AppleGridEnv()

    model = StudentNet(env.obs_size, env.n_actions)
    model.load_state_dict(torch.load("student.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Student loaded   params={n_params:,}   device={device}")
    print(f"Running {args.n} episode(s)  delay={args.delay}s  greedy={args.greedy}\n")

    totals = []
    for ep in range(1, args.n + 1):
        print(f"Episode {ep} — ", end="", flush=True)
        total = run_episode(model, env, device, args.delay, args.greedy)
        totals.append(total)
        print(f"reward = {total:.0f}")

    print(f"\nMean reward over {args.n} episodes: {np.mean(totals):.2f}")


if __name__ == "__main__":
    main()
