"""
GRPO-style reverse-KL distillation: train StudentNet via Group Relative Policy
Optimization with a teacher-approval reward bonus.

Compared to the PPO version (train_student_rl_distill.py), GRPO:

  1. Eliminates the value function / critic entirely.
  2. Runs G complete episodes per group, then normalises rewards *within the group*:

        advantage_i = (R_i − mean(group)) / (std(group) + ε)

     This makes the gradient signal scale-invariant — no kl_coef tuning needed
     to keep the bonus in the same range as env reward.

  3. Assigns that scalar episode-level advantage to every step in the episode,
     then applies the standard PPO-clip objective (no value loss).

The analogy to LLM GRPO (DeepSeek R1 style):
  prompt      → initial observation / episode start
  completion  → full episode trajectory (40 steps)
  reward      → total episode reward = Σ (r_env + kl_coef · kl_bonus)
  group       → G episodes rolled out in parallel from independent starts

Reward augmentation (same as PPO version):
  r_aug(s, a) = r_env(s, a)  +  kl_coef · [log p_teacher(a|s) − log(1/n_actions)]

Saves final student to student_grpo.pt by default.
"""

import argparse
import collections
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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


# ── student actor — no critic needed for GRPO ─────────────────────────────────
class StudentActor(nn.Module):
    """Same 2×64 trunk as StudentNet but actor-only (no critic head)."""
    def __init__(self, obs_size: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden,   hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.trunk(x))


# ── parallel group rollout ─────────────────────────────────────────────────────
@torch.no_grad()
def collect_group(
    student:    StudentActor,
    teacher:    TeacherNet,
    envs:       list,           # G AppleGridEnv instances
    kl_coef:    float,
    device,
):
    """
    Run G environments in parallel until each completes one episode.

    Returns flattened arrays for all G episodes plus per-episode stats.
    Group-relative advantages are assigned here — no value function used.
    """
    G         = len(envs)
    n_actions = envs[0].n_actions
    obs_size  = envs[0].obs_size
    log_uniform = math.log(1.0 / n_actions)   # centering baseline

    # Per-episode storage
    episodes = [
        {"obs": [], "acts": [], "logps": [], "aug_rews": [],
         "env_rew": 0.0, "kl_sum": 0.0, "n": 0}
        for _ in range(G)
    ]
    done_mask = np.zeros(G, dtype=bool)
    obs = np.stack([e.reset() for e in envs])

    student.eval()
    teacher.eval()

    while not done_mask.all():
        active_idx = np.where(~done_mask)[0]
        obs_active = obs[active_idx]
        obs_t      = torch.tensor(obs_active, dtype=torch.float32, device=device)

        # Teacher scores student actions
        teacher_logits, _ = teacher(obs_t)
        t_logp = F.log_softmax(teacher_logits, dim=-1)   # [active, n_actions]

        # Student samples
        dist  = Categorical(logits=student(obs_t))
        acts  = dist.sample()                            # [active]
        logps = dist.log_prob(acts)
        kl_b  = t_logp.gather(1, acts.unsqueeze(1)).squeeze(1) - log_uniform

        acts_np  = acts.cpu().numpy()
        logps_np = logps.cpu().numpy()
        kl_np    = kl_b.cpu().numpy()

        for j, i in enumerate(active_idx):
            obs_next, r_env, done, info = envs[i].step(acts_np[j])
            apple_rew = info.get("apple_reward", 0)
            aug_rew   = apple_rew + kl_coef * kl_np[j]

            ep = episodes[i]
            ep["obs"].append(obs_active[j].copy())
            ep["acts"].append(acts_np[j])
            ep["logps"].append(logps_np[j])
            ep["aug_rews"].append(aug_rew)
            ep["env_rew"] += apple_rew
            ep["kl_sum"]  += kl_np[j]
            ep["n"]       += 1

            if done:
                done_mask[i] = True
            else:
                obs[i] = obs_next

    # ── GRPO core: group-relative advantage normalisation ─────────────────────
    ep_totals   = np.array([sum(ep["aug_rews"]) for ep in episodes], np.float32)
    mean_r      = ep_totals.mean()
    std_r       = ep_totals.std() + 1e-8
    ep_advs     = (ep_totals - mean_r) / std_r   # [G] — scale-invariant

    # Flatten; assign episode-level advantage to every step in that episode
    all_obs, all_acts, all_logps, all_advs = [], [], [], []
    env_rews, kl_avgs = [], []

    for i, ep in enumerate(episodes):
        adv = float(ep_advs[i])
        all_obs.extend(ep["obs"])
        all_acts.extend(ep["acts"])
        all_logps.extend(ep["logps"])
        all_advs.extend([adv] * len(ep["obs"]))
        env_rews.append(ep["env_rew"])
        kl_avgs.append(ep["kl_sum"] / max(ep["n"], 1))

    return (
        np.array(all_obs,   dtype=np.float32),
        np.array(all_acts,  dtype=np.int64),
        np.array(all_logps, dtype=np.float32),
        np.array(all_advs,  dtype=np.float32),
        env_rews,
        kl_avgs,
    )


# ── policy update (no value loss) ─────────────────────────────────────────────
def grpo_update(
    student, optimizer,
    obs_np, acts_np, old_logps_np, advs_np,
    clip_eps, entropy_coef, n_epochs, mini_batch_size, max_grad_norm, device,
):
    """PPO-clip policy update with no value loss — GRPO style."""
    # Advantages already group-normalised; a second normalisation here would
    # collapse within-group signal, so we skip it.
    obs_t  = torch.tensor(obs_np,       dtype=torch.float32, device=device)
    acts_t = torch.tensor(acts_np,      dtype=torch.long,    device=device)
    olp_t  = torch.tensor(old_logps_np, dtype=torch.float32, device=device)
    adv_t  = torch.tensor(advs_np,      dtype=torch.float32, device=device)

    n = len(obs_np)
    total_loss, n_mb = 0.0, 0

    student.train()
    for _ in range(n_epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, mini_batch_size):
            idx = perm[start: start + mini_batch_size]

            dist      = Categorical(logits=student(obs_t[idx]))
            new_logps = dist.log_prob(acts_t[idx])
            entropy   = dist.entropy().mean()

            ratio = (new_logps - olp_t[idx]).exp()
            adv   = adv_t[idx]
            loss  = -torch.min(
                ratio * adv,
                ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv,
            ).mean() - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            n_mb       += 1

    return total_loss / max(n_mb, 1)


# ── eval ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_student(model, env, device, n_ep: int = 20) -> float:
    model.eval()
    total = 0.0
    for _ in range(n_ep):
        obs, done = env.reset(), False
        while not done:
            x      = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            out    = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            obs, _, done, info = env.step(Categorical(logits=logits).sample().item())
            total += info.get("apple_reward", 0)
    return total / n_ep


# ── main ──────────────────────────────────────────────────────────────────────
def train_student_grpo(
    n_updates:       int   = 300,
    group_size:      int   = 8,    # G: episodes per group for normalisation
    n_groups:        int   = 4,    # groups collected per update
    n_epochs:        int   = 1,
    mini_batch_size: int   = 256,
    lr:              float = 1e-3,
    clip_eps:        float = 0.2,
    entropy_coef:    float = 0.01,
    max_grad_norm:   float = 0.5,
    kl_coef:         float = 0.02,
    eval_every:      int   = 10,
    teacher_path:    str   = "teacher.pt",
    output_path:     str   = "student_grpo.pt",
    log_callback           = None,  # callable(env_steps, student_rew, avg_kl_bonus, entropy)
):
    device     = get_device()
    total_envs = group_size * n_groups

    env  = AppleGridEnv()
    # Create n_groups independent groups of group_size envs each
    group_envs = [
        [AppleGridEnv() for _ in range(group_size)]
        for _ in range(n_groups)
    ]

    # ── load teacher ──────────────────────────────────────────────────────────
    teacher = TeacherNet(env.obs_size, env.n_actions).to(device)
    teacher.load_state_dict(
        torch.load(teacher_path, map_location=device, weights_only=True)
    )
    teacher.eval()

    # ── build student actor (no critic) ───────────────────────────────────────
    student = StudentActor(env.obs_size, env.n_actions).to(device)
    opt     = optim.Adam(student.parameters(), lr=lr)

    n_teacher = sum(p.numel() for p in teacher.parameters())
    n_student = sum(p.numel() for p in StudentNet(env.obs_size, env.n_actions).parameters())
    eps_per_upd = group_size * n_groups

    print(f"Device         : {device}")
    print(f"Teacher params : {n_teacher:,}")
    print(f"Student params : {n_student:,}  ({100*n_student/n_teacher:.1f}% of teacher)")
    print(f"Group size     : G={group_size} episodes  ×  {n_groups} groups  =  {eps_per_upd} eps/update")
    print(f"GRPO epochs    : {n_epochs}  (no critic — advantage is group-normalised)")
    print(f"KL coef        : {kl_coef}\n")

    teacher_rew = evaluate_student(teacher, env, device, n_ep=50)
    print(f"Teacher baseline (50 ep stochastic): {teacher_rew:.2f}\n")

    header = (
        f"{'Upd':>5} | {'Episodes':>9} | {'StudRew':>8} | "
        f"{'KLbonus':>8} | {'Elapsed':>8}"
    )
    print(header)
    print("─" * len(header))

    env_rew_buf  = collections.deque(maxlen=200)
    kl_bonus_buf = collections.deque(maxlen=200)
    total_episodes = 0
    t0             = time.time()

    for upd in range(1, n_updates + 1):
        # Collect n_groups groups
        all_obs, all_acts, all_logps, all_advs = [], [], [], []

        for g in range(n_groups):
            obs_f, act_f, logp_f, adv_f, env_rews, kl_avgs = collect_group(
                student, teacher, group_envs[g], kl_coef, device
            )
            all_obs.append(obs_f);   all_acts.append(act_f)
            all_logps.append(logp_f); all_advs.append(adv_f)
            env_rew_buf.extend(env_rews)
            kl_bonus_buf.extend(kl_avgs)

        total_episodes += eps_per_upd

        obs_np  = np.concatenate(all_obs)
        acts_np = np.concatenate(all_acts)
        logp_np = np.concatenate(all_logps)
        adv_np  = np.concatenate(all_advs)

        # Quick entropy estimate
        with torch.no_grad():
            sample = obs_np[:min(512, len(obs_np))]
            ent = Categorical(
                logits=student(torch.tensor(sample, dtype=torch.float32, device=device))
            ).entropy().mean().item()

        grpo_update(
            student, opt, obs_np, acts_np, logp_np, adv_np,
            clip_eps, entropy_coef, n_epochs, mini_batch_size, max_grad_norm, device,
        )

        avg_kl  = float(np.mean(kl_bonus_buf)) if kl_bonus_buf else 0.0
        elapsed = time.time() - t0

        if upd % eval_every == 0:
            student_rew = evaluate_student(student, env, device, n_ep=20)
            print(
                f"{upd:>5} | {total_episodes:>9,} | {student_rew:>8.2f} | "
                f"{avg_kl:>8.3f} | {elapsed:>7.1f}s",
                flush=True,
            )
            if log_callback:
                log_callback(total_episodes, student_rew, avg_kl, ent)
        else:
            print(
                f"{upd:>5} | {total_episodes:>9,} | {'---':>8} | "
                f"{avg_kl:>8.3f} | {elapsed:>7.1f}s",
                flush=True,
            )

    # ── extract weights into compatible StudentNet ────────────────────────────
    student_net = StudentNet(env.obs_size, env.n_actions)
    student_net.trunk.load_state_dict(student.trunk.state_dict())
    student_net.actor.load_state_dict(student.actor.state_dict())
    torch.save(student_net.state_dict(), output_path)
    print(f"\n✓  Saved {output_path}")

    # ── final evaluation ──────────────────────────────────────────────────────
    print("\n━━━  Final evaluation  ━━━  (50 episodes, stochastic policy)")
    t_rew = evaluate_student(teacher, env, device, n_ep=50)
    s_rew = evaluate_student(student, env, device, n_ep=50)
    print(f"  Teacher : {t_rew:.2f}")
    print(f"  Student : {s_rew:.2f}  ({100*s_rew/t_rew:.1f}% of teacher reward)")
    print(f"  Params  : {n_teacher:,} → {n_student:,}  ({100*n_student/n_teacher:.1f}%)")
    print(f"  Total time : {time.time() - t0:.1f}s")

    return student


if __name__ == "__main__":
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    p = argparse.ArgumentParser()
    p.add_argument("--updates",    type=int,   default=300,
                   help="Number of GRPO update steps")
    p.add_argument("--group-size", type=int,   default=8,
                   help="G: episodes per group for reward normalisation")
    p.add_argument("--n-groups",   type=int,   default=4,
                   help="Groups collected per update (total = G × n_groups eps)")
    p.add_argument("--epochs",     type=int,   default=1,
                   help="Policy gradient epochs per update (no critic)")
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--kl-coef",    type=float, default=0.02,
                   help="Weight on centred log-likelihood-ratio reward bonus")
    p.add_argument("--teacher",    type=str,   default="teacher.pt")
    p.add_argument("--output",     type=str,   default="student_grpo.pt")
    args = p.parse_args()

    train_student_grpo(
        n_updates    = args.updates,
        group_size   = args.group_size,
        n_groups     = args.n_groups,
        n_epochs     = args.epochs,
        lr           = args.lr,
        kl_coef      = args.kl_coef,
        teacher_path = args.teacher,
        output_path  = args.output,
    )
