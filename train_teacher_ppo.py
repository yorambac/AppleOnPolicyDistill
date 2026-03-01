"""
Train a PPO teacher on AppleGridEnv.

Algorithm
─────────
Each update collects a fixed-length rollout of `n_steps` steps from every
parallel environment (batch = n_steps × n_envs transitions), then performs
`n_epochs` passes over that batch in random mini-batches.

Key differences from A2C:
  • Clipped surrogate objective  —  ratio clipped to [1−ε, 1+ε]
  • GAE advantages               —  λ-weighted multi-step returns
  • Multiple epochs per rollout  —  sample re-use without catastrophic drift
  • Entropy bonus (0.01)         —  mild exploration regularisation

Reward shaping: same potential-based schedule as the A2C teacher —
linearly annealed from 0.05 → 0 over the first `anneal_frac` of training.

Saves weights to `teacher_ppo.pt` by default.
"""

import argparse
import collections
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from grid_env import AppleGridEnv
from models import TeacherNet

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("MacOSX")
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
    """Greedy nearest-apple oracle."""
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


# ── VecEnvPPO: persistent state, auto-reset on done ──────────────────────────
class VecEnvPPO:
    """
    Unlike the A2C VecEnv, this one keeps its observations alive across
    updates (no reset at the start of each rollout).  When an episode ends
    the env resets internally so the NEXT observation is the initial state
    of the new episode — the episode boundary is recorded via done=1.
    """
    def __init__(self, n: int):
        self.envs = [AppleGridEnv() for _ in range(n)]
        self.n    = n
        self.obs  = np.stack([e.reset() for e in self.envs])

    def step(self, actions: np.ndarray):
        rews, dones, infos = [], [], []
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            obs, r, done, info = env.step(int(a))
            if done:
                obs = env.reset()
            self.obs[i] = obs
            rews.append(r)
            dones.append(float(done))
            infos.append(info)
        return (self.obs.copy(),
                np.array(rews,  dtype=np.float32),
                np.array(dones, dtype=np.float32),
                infos)

    def set_shaping_coef(self, coef: float):
        for e in self.envs:
            e.shaping_coef = coef


# ── rollout collection + GAE ───────────────────────────────────────────────────
@torch.no_grad()
def collect_rollout(model, venv: VecEnvPPO, n_steps: int, device, gamma: float, gae_lambda: float):
    """
    Collect exactly n_steps steps from every env, then compute GAE advantages.

    Returns (all flattened to shape [n_steps * n_envs]):
        obs_f, act_f, logp_f, adv_f, ret_f  — training tensors
        completed_apples                      — list of per-episode apple counts
    """
    n_envs   = venv.n
    obs_size = venv.envs[0].obs_size

    obs_buf  = np.empty((n_steps, n_envs, obs_size), np.float32)
    act_buf  = np.empty((n_steps, n_envs),           np.int64)
    logp_buf = np.empty((n_steps, n_envs),           np.float32)
    val_buf  = np.empty((n_steps, n_envs),           np.float32)
    rew_buf  = np.empty((n_steps, n_envs),           np.float32)
    done_buf = np.empty((n_steps, n_envs),           np.float32)

    ep_apple       = np.zeros(n_envs, np.float32)
    completed_apples = []

    model.eval()
    obs = venv.obs.copy()

    for t in range(n_steps):
        obs_t          = torch.tensor(obs, dtype=torch.float32, device=device)
        logits, vals   = model(obs_t)
        dist           = Categorical(logits=logits)
        acts           = dist.sample()
        logps          = dist.log_prob(acts)

        obs_buf[t]  = obs
        act_buf[t]  = acts.cpu().numpy()
        logp_buf[t] = logps.cpu().numpy()
        val_buf[t]  = vals.cpu().numpy()

        obs, rews, dones, infos = venv.step(acts.cpu().numpy())

        rew_buf[t]  = rews
        done_buf[t] = dones

        ep_apple += np.array([info.get("apple_reward", 0) for info in infos])
        for i in range(n_envs):
            if dones[i]:
                completed_apples.append(ep_apple[i])
                ep_apple[i] = 0.0

    # Bootstrap value at the end of the rollout
    _, last_vals = model(torch.tensor(venv.obs, dtype=torch.float32, device=device))
    last_vals = last_vals.cpu().numpy()

    # GAE (reverse scan)
    # δ_t = r_t + γ * V(s_{t+1}) * (1−done_t) − V(s_t)
    # Â_t = δ_t + γλ * (1−done_t) * Â_{t+1}
    advantages = np.zeros((n_steps, n_envs), np.float32)
    gae        = np.zeros(n_envs, np.float32)

    for t in reversed(range(n_steps)):
        next_val = val_buf[t + 1] if t < n_steps - 1 else last_vals
        not_done = 1.0 - done_buf[t]
        delta    = rew_buf[t] + gamma * next_val * not_done - val_buf[t]
        gae      = delta + gamma * gae_lambda * not_done * gae
        advantages[t] = gae

    returns = advantages + val_buf

    # Flatten: (n_steps * n_envs, ...)
    obs_f  = obs_buf.reshape(-1, obs_size)
    act_f  = act_buf.reshape(-1)
    logp_f = logp_buf.reshape(-1)
    adv_f  = advantages.reshape(-1)
    ret_f  = returns.reshape(-1)

    return obs_f, act_f, logp_f, adv_f, ret_f, completed_apples


# ── PPO update ────────────────────────────────────────────────────────────────
def ppo_update(
    model, optimizer,
    obs_np, acts_np, old_logps_np, advs_np, rets_np,
    clip_eps, value_coef, entropy_coef,
    n_epochs, mini_batch_size, max_grad_norm, device,
):
    """
    n_epochs passes of random mini-batches over the collected rollout.
    Normalises advantages once over the full batch before any mini-batch split.
    Returns: (mean_loss, mean_clip_frac, mean_approx_kl)
    """
    # Normalise advantages over the full batch
    advs_np = (advs_np - advs_np.mean()) / (advs_np.std() + 1e-8)

    obs_t  = torch.tensor(obs_np,      dtype=torch.float32, device=device)
    acts_t = torch.tensor(acts_np,     dtype=torch.long,    device=device)
    olp_t  = torch.tensor(old_logps_np, dtype=torch.float32, device=device)
    adv_t  = torch.tensor(advs_np,     dtype=torch.float32, device=device)
    ret_t  = torch.tensor(rets_np,     dtype=torch.float32, device=device)

    n = len(obs_np)
    total_loss = clip_frac = approx_kl = 0.0
    n_mb = 0

    model.train()
    for _ in range(n_epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, mini_batch_size):
            idx = perm[start: start + mini_batch_size]

            logits, vals  = model(obs_t[idx])
            dist          = Categorical(logits=logits)
            new_logps     = dist.log_prob(acts_t[idx])
            entropy       = dist.entropy().mean()

            ratio  = (new_logps - olp_t[idx]).exp()
            adv_mb = adv_t[idx]
            surr1  = ratio * adv_mb
            surr2  = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss  = F.mse_loss(vals, ret_t[idx])
            loss        = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                clip_frac  += ((ratio - 1).abs() > clip_eps).float().mean().item()
                approx_kl  += (olp_t[idx] - new_logps).mean().item()
                total_loss += loss.item()
            n_mb += 1

    return total_loss / n_mb, clip_frac / n_mb, approx_kl / n_mb


# ── live plot ─────────────────────────────────────────────────────────────────
class LivePlot:
    def __init__(self, rand_rew, oracle_rew, n_updates, n_envs):
        if not HAS_MPL:
            self.ok = False
            return
        try:
            plt.ion()
            self.fig = plt.figure(figsize=(13, 7), facecolor="#0d1117")
            self.fig.suptitle("Teacher Training  —  PPO + GAE + Potential Shaping",
                              color="#58a6ff", fontsize=12, fontweight="bold")
            gs = mgridspec.GridSpec(2, 3, figure=self.fig,
                                    hspace=0.45, wspace=0.4,
                                    left=0.07, right=0.97, top=0.91, bottom=0.09)
            self.ax_rew  = self.fig.add_subplot(gs[:, :2])
            self.ax_ent  = self.fig.add_subplot(gs[0, 2])
            self.ax_loss = self.fig.add_subplot(gs[1, 2])

            for ax in [self.ax_rew, self.ax_ent, self.ax_loss]:
                ax.set_facecolor("#161b22")
                for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
                ax.tick_params(colors="#8b949e")
                ax.yaxis.grid(True, alpha=0.25, color="#30363d")

            self.ax_rew.set_title("Rolling Apple Reward (last 500 ep)", color="white", fontsize=10)
            self.ax_rew.set_xlabel("Episodes", color="#8b949e")
            self.ax_rew.set_ylabel("Apples / episode", color="#8b949e")
            self.ax_rew.axhline(rand_rew,   color="#ffa657", lw=1.5, ls="--", label=f"Random  {rand_rew:.1f}")
            self.ax_rew.axhline(oracle_rew, color="#3fb950", lw=1.5, ls="--", label=f"Oracle  {oracle_rew:.1f}")
            self.rew_line, = self.ax_rew.plot([], [], color="#58a6ff", lw=2, label="Trained (PPO)")
            self.ax_rew.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
            self.ax_rew.set_xlim(0, n_updates * n_envs * 8)   # approx ep count
            self.ax_rew.set_ylim(0, oracle_rew * 1.05)

            self.ax_ent.set_title("Entropy (nats)", color="white", fontsize=9)
            self.ax_ent.axhline(np.log(4), color="#ffa657", lw=1, ls="--")
            self.ent_line, = self.ax_ent.plot([], [], color="#bc8cff", lw=1.5)
            self.ax_ent.set_ylim(0, np.log(4) * 1.1)
            self.ax_ent.set_xlabel("Episodes", color="#8b949e", fontsize=8)

            self.ax_loss.set_title("Value Loss + Clip Fraction", color="white", fontsize=9)
            self.loss_line, = self.ax_loss.plot([], [], color="#f85149", lw=1.5, label="Value loss")
            self.ax_loss2   = self.ax_loss.twinx()
            self.ax_loss2.set_facecolor("#161b22")
            self.ax_loss2.tick_params(colors="#8b949e")
            self.clip_line, = self.ax_loss2.plot([], [], color="#ffa657", lw=1.5, ls=":", label="Clip frac")
            self.ax_loss.set_xlabel("Episodes", color="#8b949e", fontsize=8)
            self.ax_loss.set_ylabel("Loss", color="#f85149", fontsize=8)
            self.ax_loss2.set_ylabel("Clip fraction", color="#ffa657", fontsize=8)
            self.ax_loss2.set_ylim(0, 0.5)

            self.xs, self.rews, self.ents, self.losses, self.clips = [], [], [], [], []
            plt.show(block=False)
            self.ok = True
        except Exception as e:
            print(f"[LivePlot] disabled: {e}")
            self.ok = False

    def update(self, ep, rew, ent, loss, clip_frac):
        if not self.ok:
            return
        try:
            self.xs.append(ep);    self.rews.append(rew)
            self.ents.append(ent); self.losses.append(loss); self.clips.append(clip_frac)

            self.rew_line.set_data(self.xs, self.rews)
            self.ent_line.set_data(self.xs, self.ents)
            self.loss_line.set_data(self.xs, self.losses)
            self.clip_line.set_data(self.xs, self.clips)

            for ax in [self.ax_ent, self.ax_loss]:
                ax.relim(); ax.autoscale_view(scaley=True)
            self.ax_loss2.relim(); self.ax_loss2.autoscale_view(scaley=False)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass


# ── training ──────────────────────────────────────────────────────────────────
def train(
    total_timesteps: int   = 2_500_000,
    n_envs:          int   = 16,
    n_steps:         int   = 128,
    n_epochs:        int   = 4,
    mini_batch_size: int   = 256,
    gamma:           float = 0.99,
    gae_lambda:      float = 0.95,
    lr:              float = 3e-4,
    clip_eps:        float = 0.2,
    entropy_coef:    float = 0.01,
    value_coef:      float = 0.5,
    max_grad_norm:   float = 0.5,
    shaping_init:    float = 0.05,
    anneal_frac:     float = 0.75,
    log_every:       int   = 10,
    use_wandb:       bool  = True,
    live_plot:       bool  = True,
    output:          str   = "teacher_ppo.pt",
    log_callback     = None,   # callable(ep, apples, entropy, value_loss, clip_frac)
):
    device     = get_device()
    batch_size = n_steps * n_envs                    # transitions per update
    n_updates  = total_timesteps // batch_size

    env1    = AppleGridEnv()
    venv    = VecEnvPPO(n_envs)
    model   = TeacherNet(env1.obs_size, env1.n_actions).to(device)
    opt     = optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"Device       : {device}")
    print(f"TeacherNet   obs={env1.obs_size}  params={n_params:,}")
    print(f"Batch size   : {n_steps} steps × {n_envs} envs = {batch_size:,} transitions")
    print(f"Updates      : {n_updates:,}  ({n_epochs} PPO epochs each,  mini-batch {mini_batch_size})")
    print(f"Total steps  : {n_updates * batch_size:,}")
    print(f"Shaping      : {shaping_init:.3f} → 0 over first {anneal_frac*100:.0f}% of training\n")

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
            name=f"ppo-{total_timesteps//1_000_000}Msteps",
            config=dict(total_timesteps=total_timesteps, n_envs=n_envs, n_steps=n_steps,
                        n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda,
                        lr=lr, clip_eps=clip_eps, entropy_coef=entropy_coef,
                        shaping_init=shaping_init, anneal_frac=anneal_frac),
        )
        print(f"W&B run: {wb.url}\n")
    elif use_wandb:
        print("W&B not available (pip install wandb && wandb login)\n")

    # ── live plot ─────────────────────────────────────────────────────────────
    plot = LivePlot(rand_rew, oracle_rew, n_updates, n_envs) if live_plot else None

    # ── header ────────────────────────────────────────────────────────────────
    hdr = (f"{'Upd':>5} | {'Episodes':>8} | {'Apples':>6} | "
           f"{'Entropy':>7} | {'ValLoss':>7} | {'ClipFrac':>8} | "
           f"{'KL':>6} | {'ShpCoef':>7} | {'Time':>6}")
    print(hdr)
    print("─" * len(hdr))

    # ── rolling buffers ───────────────────────────────────────────────────────
    apple_buf = collections.deque(maxlen=WINDOW)
    ent_buf   = collections.deque(maxlen=log_every)
    loss_buf  = collections.deque(maxlen=log_every)
    clip_buf  = collections.deque(maxlen=log_every)
    kl_buf    = collections.deque(maxlen=log_every)
    total_ep  = 0
    t0        = time.time()

    for upd in range(1, n_updates + 1):
        timestep = upd * batch_size

        # ── anneal shaping coef ───────────────────────────────────────────────
        shp_coef = shaping_init * max(0.0, 1.0 - timestep / (total_timesteps * anneal_frac))
        venv.set_shaping_coef(shp_coef)

        # ── collect rollout + GAE ─────────────────────────────────────────────
        obs_f, act_f, logp_f, adv_f, ret_f, completed = collect_rollout(
            model, venv, n_steps, device, gamma, gae_lambda
        )
        apple_buf.extend(completed)
        total_ep += len(completed)

        # ── entropy (from old logits, cheap) ──────────────────────────────────
        with torch.no_grad():
            logits_sample, _ = model(
                torch.tensor(obs_f[:min(512, len(obs_f))], dtype=torch.float32, device=device)
            )
            ent = Categorical(logits=logits_sample).entropy().mean().item()

        # ── PPO update ────────────────────────────────────────────────────────
        loss, clip_frac, approx_kl = ppo_update(
            model, opt,
            obs_f, act_f, logp_f, adv_f, ret_f,
            clip_eps, value_coef, entropy_coef,
            n_epochs, mini_batch_size, max_grad_norm, device,
        )

        ent_buf.append(ent)
        loss_buf.append(loss)
        clip_buf.append(clip_frac)
        kl_buf.append(approx_kl)

        # ── log ───────────────────────────────────────────────────────────────
        if upd % log_every == 0 and len(apple_buf) > 0:
            avg_apples = float(np.mean(apple_buf))
            avg_ent    = float(np.mean(ent_buf))
            avg_loss   = float(np.mean(loss_buf))
            avg_clip   = float(np.mean(clip_buf))
            avg_kl     = float(np.mean(kl_buf))
            elapsed    = time.time() - t0

            print(
                f"{upd:>5} | {total_ep:>8,} | {avg_apples:>6.2f} | "
                f"{avg_ent:>7.3f} | {avg_loss:>7.4f} | {avg_clip:>8.3f} | "
                f"{avg_kl:>6.4f} | {shp_coef:>7.4f} | {elapsed:>5.0f}s",
                flush=True,
            )

            if wb:
                wb.log({"apples": avg_apples, "entropy": avg_ent,
                        "value_loss": avg_loss, "clip_frac": avg_clip,
                        "approx_kl": avg_kl, "shaping_coef": shp_coef,
                        "episodes": total_ep, "timesteps": timestep,
                        "elapsed": elapsed})

            if plot:
                plot.update(total_ep, avg_apples, avg_ent, avg_loss, avg_clip)

            if log_callback:
                log_callback(total_ep, avg_apples, avg_ent, avg_loss, avg_clip)

    # ── final eval ────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), output)
    elapsed = time.time() - t0

    final = evaluate(model, env1, device, n_ep=200)
    gap   = oracle_rew - rand_rew

    print(f"\n{'─'*60}")
    print(f"  Random oracle  : {rand_rew:.2f}")
    print(f"  Greedy oracle  : {oracle_rew:.2f}")
    print(f"  Trained policy : {final:.2f}  ({100*(final-rand_rew)/gap:.0f}% of oracle gap)")
    print(f"  Total time     : {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"  Saved → {output}")

    if wb:
        wb.summary["final_apples"] = final
        wb.summary["pct_oracle"]   = 100 * (final - rand_rew) / gap
        wb.finish()

    if plot and plot.ok:
        print("\nClose the plot window to exit.")
        plt.ioff()
        plt.show()

    return model


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int,   default=2_500_000)
    p.add_argument("--envs",      type=int,   default=16)
    p.add_argument("--steps",     type=int,   default=128,  help="n_steps per env per rollout")
    p.add_argument("--epochs",    type=int,   default=4,    help="PPO re-use epochs")
    p.add_argument("--lr",        type=float, default=3e-4)
    p.add_argument("--clip",      type=float, default=0.2,  help="PPO clip epsilon")
    p.add_argument("--no-wandb",  action="store_true")
    p.add_argument("--no-plot",   action="store_true")
    p.add_argument("--output",    type=str,   default="teacher_ppo.pt")
    args = p.parse_args()

    train(
        total_timesteps = args.timesteps,
        n_envs          = args.envs,
        n_steps         = args.steps,
        n_epochs        = args.epochs,
        lr              = args.lr,
        clip_eps        = args.clip,
        use_wandb       = not args.no_wandb,
        live_plot       = not args.no_plot,
        output          = args.output,
    )
