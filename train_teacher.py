"""
Train an A2C teacher on AppleGridEnv.

Features
────────
• Potential-based shaping reward, linearly annealed to 0 over the first
  `anneal_frac` of training so the policy eventually relies only on apples.
• Live matplotlib training plot updated every `log_every` steps.
• Optional Weights & Biases logging (--no-wandb to disable).
• Diagnostic baselines (random + greedy oracle) printed at startup.
• Rolling-500-episode apple count, entropy, critic loss, grad norm logged.
"""

import argparse
import collections
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from grid_env import AppleGridEnv
from models import TeacherNet

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("MacOSX")          # non-blocking on macOS; change if needed
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as mgridspec
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

WINDOW = 500   # rolling reward window (episodes)


# ── device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── helpers ───────────────────────────────────────────────────────────────────
def mc_returns(rewards, gamma):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return out


# ── baselines ─────────────────────────────────────────────────────────────────
def random_baseline(env, n_ep=200):
    total = 0.0
    for _ in range(n_ep):
        env.reset()
        done = False
        while not done:
            _, _, done, info = env.step(np.random.randint(4))
            total += info.get("apple_reward", 0)
    return total / n_ep


def oracle_baseline(env, n_ep=200):
    """Greedy nearest-apple oracle (upper-bound on what a learned policy can do)."""
    total = 0.0
    for _ in range(n_ep):
        env.reset()
        done = False
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
            total += info.get("apple_reward", 0)
    return total / n_ep


# ── eval ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, env, device, n_ep=100):
    model.eval()
    total = 0.0
    for _ in range(n_ep):
        obs, done = env.reset(), False
        while not done:
            x         = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(x)
            obs, _, done, info = env.step(Categorical(logits=logits).sample().item())
            total += info.get("apple_reward", 0)
    model.train()
    return total / n_ep


@torch.no_grad()
def action_dist_check(model, env, device, n_ep=20):
    model.eval()
    counts = np.zeros(4, int)
    for _ in range(n_ep):
        obs, done = env.reset(), False
        while not done:
            x         = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(x)
            a = Categorical(logits=logits).sample().item()
            counts[a] += 1
            obs, _, done, _ = env.step(a)
    model.train()
    p = 100 * counts / counts.sum()
    return {s: f"{v:.1f}%" for s, v in zip(["↑","↓","←","→"], p)}


# ── VecEnv ────────────────────────────────────────────────────────────────────
class VecEnv:
    def __init__(self, n):
        self.envs = [AppleGridEnv() for _ in range(n)]
        self.n    = n

    def reset(self):
        return np.stack([e.reset() for e in self.envs])

    def set_shaping_coef(self, coef):
        for e in self.envs:
            e.shaping_coef = coef


# ── live plot ─────────────────────────────────────────────────────────────────
class LivePlot:
    def __init__(self, rand_rew, oracle_rew, n_updates, n_envs):
        if not HAS_MPL:
            self.ok = False
            return
        try:
            plt.ion()
            self.fig = plt.figure(figsize=(13, 7), facecolor="#0d1117")
            self.fig.suptitle("Teacher Training  —  A2C + Potential Shaping",
                              color="#58a6ff", fontsize=12, fontweight="bold")
            gs = mgridspec.GridSpec(2, 3, figure=self.fig,
                                    hspace=0.45, wspace=0.4,
                                    left=0.07, right=0.97, top=0.91, bottom=0.09)
            self.ax_rew  = self.fig.add_subplot(gs[:, :2])   # wide left
            self.ax_ent  = self.fig.add_subplot(gs[0, 2])
            self.ax_loss = self.fig.add_subplot(gs[1, 2])

            for ax in [self.ax_rew, self.ax_ent, self.ax_loss]:
                ax.set_facecolor("#161b22")
                for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
                ax.tick_params(colors="#8b949e")
                ax.yaxis.grid(True, alpha=0.25, color="#30363d")

            # reward axes
            self.ax_rew.set_title("Rolling Apple Reward (last 500 ep)", color="white", fontsize=10)
            self.ax_rew.set_xlabel("Episodes", color="#8b949e")
            self.ax_rew.set_ylabel("Apples / episode", color="#8b949e")
            self.ax_rew.axhline(rand_rew,   color="#ffa657", lw=1.5, ls="--", label=f"Random  {rand_rew:.1f}")
            self.ax_rew.axhline(oracle_rew, color="#3fb950", lw=1.5, ls="--", label=f"Oracle  {oracle_rew:.1f}")
            self.rew_line, = self.ax_rew.plot([], [], color="#58a6ff", lw=2, label="Trained")
            self.ax_rew.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
            self.ax_rew.set_xlim(0, n_updates * n_envs)
            self.ax_rew.set_ylim(0, oracle_rew * 1.05)

            # entropy axes
            self.ax_ent.set_title("Entropy (nats)", color="white", fontsize=9)
            self.ax_ent.axhline(np.log(4), color="#ffa657", lw=1, ls="--")
            self.ent_line, = self.ax_ent.plot([], [], color="#bc8cff", lw=1.5)
            self.ax_ent.set_ylim(0, np.log(4) * 1.1)
            self.ax_ent.set_xlabel("Episodes", color="#8b949e", fontsize=8)

            # loss axes (shaping coef on secondary y)
            self.ax_loss.set_title("Critic Loss + Shaping Coef", color="white", fontsize=9)
            self.crit_line, = self.ax_loss.plot([], [], color="#f85149", lw=1.5, label="Critic loss")
            self.ax_loss2   = self.ax_loss.twinx()
            self.ax_loss2.set_facecolor("#161b22")
            self.ax_loss2.tick_params(colors="#8b949e")
            self.shp_line,  = self.ax_loss2.plot([], [], color="#ffa657", lw=1.5, ls=":", label="Shaping coef")
            self.ax_loss.set_xlabel("Episodes", color="#8b949e", fontsize=8)
            self.ax_loss.set_ylabel("Loss", color="#f85149", fontsize=8)
            self.ax_loss2.set_ylabel("Shaping coef", color="#ffa657", fontsize=8)
            self.ax_loss2.set_ylim(0, 0.06)

            self.xs, self.rews, self.ents, self.crits, self.shps = [], [], [], [], []
            plt.show(block=False)
            self.ok = True
        except Exception as e:
            print(f"[LivePlot] disabled: {e}")
            self.ok = False

    def update(self, ep, rew, ent, crit, shp_coef):
        if not self.ok:
            return
        try:
            self.xs.append(ep)
            self.rews.append(rew)
            self.ents.append(ent)
            self.crits.append(crit)
            self.shps.append(shp_coef)

            self.rew_line.set_data(self.xs, self.rews)
            self.ent_line.set_data(self.xs, self.ents)
            self.crit_line.set_data(self.xs, self.crits)
            self.shp_line.set_data(self.xs, self.shps)

            for ax in [self.ax_ent, self.ax_loss]:
                ax.relim(); ax.autoscale_view(scaley=True)
            self.ax_loss2.relim(); self.ax_loss2.autoscale_view(scaley=False)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass   # never crash training because of plot


# ── training ──────────────────────────────────────────────────────────────────
def train(
    n_updates:    int   = 4000,
    n_envs:       int   = 16,
    gamma:        float = 0.99,
    lr:           float = 3e-4,
    entropy_coef: float = 0.0,
    value_coef:   float = 0.5,
    max_grad_norm:float = 0.5,
    shaping_init: float = 0.05,   # initial shaping scale
    anneal_frac:  float = 0.75,   # fraction of training over which shaping → 0
    log_every:    int   = 10,
    use_wandb:    bool  = True,
    live_plot:    bool  = True,
    output:       str   = "teacher.pt",
    log_callback  = None,   # callable(ep, apples, entropy, critic_loss, shaping_coef)
):
    device  = get_device()
    env1    = AppleGridEnv()
    venv    = VecEnv(n_envs)
    model   = TeacherNet(env1.obs_size, env1.n_actions).to(device)
    opt     = optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"Device : {device}")
    print(f"TeacherNet  obs={env1.obs_size}  params={n_params:,}")
    print(f"Training  {n_updates} updates × {n_envs} envs = {n_updates*n_envs:,} episodes")
    print(f"Shaping   {shaping_init:.3f} → 0 over first {anneal_frac*100:.0f}% of training\n")

    # ── baselines ─────────────────────────────────────────────────────────────
    print("Computing baselines (200 ep each)…", flush=True)
    rand_rew   = random_baseline(env1)
    oracle_rew = oracle_baseline(env1)
    print(f"  Random oracle  : {rand_rew:.2f}")
    print(f"  Greedy oracle  : {oracle_rew:.2f}  ← target")
    print(f"  Uniform entropy: {np.log(4):.3f} nats\n")

    # ── W&B ───────────────────────────────────────────────────────────────────
    wb = None
    if use_wandb and HAS_WANDB:
        wb = wandb.init(
            project="apple-grid-distillation",
            name=f"teacher-{n_updates}upd",
            config=dict(n_updates=n_updates, n_envs=n_envs, gamma=gamma, lr=lr,
                        shaping_init=shaping_init, anneal_frac=anneal_frac,
                        rand_rew=rand_rew, oracle_rew=oracle_rew),
        )
        print(f"W&B run: {wb.url}\n")
    elif use_wandb:
        print("W&B not available (run: pip install wandb && wandb login)\n")

    # ── live plot ─────────────────────────────────────────────────────────────
    plot = LivePlot(rand_rew, oracle_rew, n_updates, n_envs) if live_plot else None

    # ── header ────────────────────────────────────────────────────────────────
    hdr = (f"{'Upd':>5} | {'Episodes':>8} | "
           f"{'Apples':>6} | {'Entropy':>7} | {'CritLoss':>8} | "
           f"{'GradNorm':>8} | {'ShpCoef':>7} | {'Time':>6}")
    print(hdr)
    print("─" * len(hdr))

    # ── rolling buffers ───────────────────────────────────────────────────────
    apple_buf = collections.deque(maxlen=WINDOW)
    ent_buf   = collections.deque(maxlen=log_every)
    crit_buf  = collections.deque(maxlen=log_every)
    grad_buf  = collections.deque(maxlen=log_every)
    t0        = time.time()

    for upd in range(1, n_updates + 1):
        # ── anneal shaping coef ───────────────────────────────────────────────
        anneal_steps = n_updates * anneal_frac
        shp_coef     = shaping_init * max(0.0, 1.0 - upd / anneal_steps)
        venv.set_shaping_coef(shp_coef)
        env1.shaping_coef = shp_coef

        # ── parallel rollout ──────────────────────────────────────────────────
        obs_batch = venv.reset()
        active    = np.ones(n_envs, bool)

        ep_log_probs = [[] for _ in range(n_envs)]
        ep_values    = [[] for _ in range(n_envs)]
        ep_entropies = [[] for _ in range(n_envs)]
        ep_rewards   = [[] for _ in range(n_envs)]
        ep_apples    = np.zeros(n_envs)

        while active.any():
            idx    = np.where(active)[0]
            obs_t  = torch.tensor(obs_batch[idx], dtype=torch.float32, device=device)
            logits, vals = model(obs_t)
            dist   = Categorical(logits=logits)
            acts   = dist.sample()
            lp_a   = dist.log_prob(acts)
            ent_a  = dist.entropy()
            nxt    = obs_batch.copy()

            for li, ei in enumerate(idx):
                obs_n, r, done, info = venv.envs[ei].step(int(acts[li].item()))
                nxt[ei] = obs_n
                ep_log_probs[ei].append(lp_a[li])
                ep_values[ei].append(vals[li])
                ep_entropies[ei].append(ent_a[li])
                ep_rewards[ei].append(r)
                ep_apples[ei] += info.get("apple_reward", 0)
                if done:
                    active[ei] = False

            obs_batch = nxt

        # ── losses ────────────────────────────────────────────────────────────
        all_actor, all_critic, all_ent = [], [], []

        for i in range(n_envs):
            ret_i = torch.tensor(mc_returns(ep_rewards[i], gamma), dtype=torch.float32, device=device)
            val_i = torch.stack(ep_values[i])
            lp_i  = torch.stack(ep_log_probs[i])
            ent_i = torch.stack(ep_entropies[i]).mean()

            adv_i = ret_i - val_i.detach()
            adv_i = (adv_i - adv_i.mean()) / (adv_i.std() + 1e-8)

            all_actor.append(-(lp_i * adv_i).mean())
            all_critic.append(nn.functional.mse_loss(val_i, ret_i))
            all_ent.append(ent_i)
            apple_buf.append(ep_apples[i])

        actor_loss  = torch.stack(all_actor).mean()
        critic_loss = torch.stack(all_critic).mean()
        entropy     = torch.stack(all_ent).mean()
        loss        = actor_loss + value_coef * critic_loss - entropy_coef * entropy

        opt.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()

        ent_buf.append(entropy.item())
        crit_buf.append(critic_loss.item())
        grad_buf.append(float(grad_norm))

        # ── log ───────────────────────────────────────────────────────────────
        if upd % log_every == 0:
            avg_apples = float(np.mean(apple_buf))
            avg_ent    = float(np.mean(ent_buf))
            avg_crit   = float(np.mean(crit_buf))
            avg_grad   = float(np.mean(grad_buf))
            ep_done    = upd * n_envs
            elapsed    = time.time() - t0

            print(
                f"{upd:>5} | {ep_done:>8,} | "
                f"{avg_apples:>6.2f} | {avg_ent:>7.3f} | {avg_crit:>8.4f} | "
                f"{avg_grad:>8.4f} | {shp_coef:>7.4f} | {elapsed:>5.0f}s",
                flush=True,
            )

            if wb:
                wb.log({"apples": avg_apples, "entropy": avg_ent,
                        "critic_loss": avg_crit, "grad_norm": avg_grad,
                        "shaping_coef": shp_coef,
                        "episodes": ep_done, "elapsed": elapsed})

            if plot:
                plot.update(ep_done, avg_apples, avg_ent, avg_crit, shp_coef)

            if log_callback:
                log_callback(ep_done, avg_apples, avg_ent, avg_crit, shp_coef)

    # ── final ─────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), output)
    elapsed = time.time() - t0

    final = evaluate(model, env1, device, n_ep=200)
    adist = action_dist_check(model, env1, device)
    gap   = oracle_rew - rand_rew

    print(f"\n{'─'*60}")
    print(f"  Random oracle  : {rand_rew:.2f}")
    print(f"  Greedy oracle  : {oracle_rew:.2f}")
    print(f"  Trained policy : {final:.2f}  ({100*(final-rand_rew)/gap:.0f}% of oracle gap)")
    print(f"  Action dist    : {adist}  (uniform=25%)")
    print(f"  Total time     : {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"  Saved → {output}")

    if wb:
        wb.summary["final_apples"] = final
        wb.summary["pct_oracle"]   = 100*(final-rand_rew)/gap
        wb.finish()

    if plot and plot.ok:
        print("\nClose the plot window to exit.")
        plt.ioff()
        plt.show()

    return model


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--updates",  type=int,   default=4000)
    p.add_argument("--envs",     type=int,   default=16)
    p.add_argument("--lr",       type=float, default=3e-4)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-plot",  action="store_true")
    p.add_argument("--output",   type=str, default="teacher.pt")
    args = p.parse_args()

    train(
        n_updates=args.updates,
        n_envs=args.envs,
        lr=args.lr,
        use_wandb=not args.no_wandb,
        live_plot=not args.no_plot,
        output=args.output,
    )
