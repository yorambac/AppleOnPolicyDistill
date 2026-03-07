"""
Reverse-KL RL distillation: train StudentNet via PPO with a teacher-approval
reward bonus, approximating minimisation of KL( p_student ‖ p_teacher ).

This is the *reverse* KL direction — the student's own state distribution
drives training.  Rollouts are collected under the *student* policy, and the
student is trained with PPO using an augmented reward:

    r_aug(s, a) = r_env(s, a)  +  kl_coef · log p_teacher(a | s)

where a is sampled from the student and r_env is the sparse apple reward.

Interpretation
──────────────
KL( p_student ‖ p_teacher ) = E_{a ~ student}[ log p_student(a|s) − log p_teacher(a|s) ]

To minimise this, the student should:
  • Maximise  E_{a ~ student}[ log p_teacher(a|s) ]  ← the reward bonus above
  • Minimise  E_{a ~ student}[ log p_student(a|s) ]  ← entropy maximisation
    (handled automatically by PPO's entropy coefficient)

Because the student generates its own rollouts, it can explore states not
visited by the teacher, potentially recovering behaviour the forward-KL
method (train_student_logit_distill.py) would miss in novel regions.

Architecture
────────────
A StudentActorCritic (same 2×64 trunk as StudentNet, plus a critic head) is
used during PPO training.  At the end, actor weights are extracted into a
standard StudentNet for compatibility with compare.py / visualize_student.py.

Saves the final student to student_rl.pt by default.
"""

import argparse
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


# ── student actor-critic for PPO ──────────────────────────────────────────────
class StudentActorCritic(nn.Module):
    """
    Same 2×64 trunk as StudentNet but with an added critic head.
    Used only during RL distillation; actor weights are extracted at the end.
    """
    def __init__(self, obs_size: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden,   hidden), nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)


# ── persistent vec-env (auto-reset, no shaping) ───────────────────────────────
class VecEnvStudent:
    """Persistent-state vec-env for student PPO (no shaping)."""
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


# ── rollout collection with augmented reward + GAE ────────────────────────────
@torch.no_grad()
def collect_rollout(
    student_ac: StudentActorCritic,
    teacher:    TeacherNet,
    venv:       VecEnvStudent,
    n_steps:    int,
    kl_coef:    float,
    device,
    gamma:      float,
    gae_lambda: float,
):
    """
    Collect n_steps steps from every env under the student policy.
    Reward is augmented with kl_coef * log p_teacher(a_student | s).

    Returns (all flattened to [n_steps * n_envs]):
        obs_f, act_f, logp_f, adv_f, ret_f  — PPO training arrays
        completed_env_rew                    — list of per-episode apple counts
        completed_kl_bonus                   — list of per-episode mean kl bonus
    """
    n_envs   = venv.n
    obs_size = venv.envs[0].obs_size

    obs_buf     = np.empty((n_steps, n_envs, obs_size), np.float32)
    act_buf     = np.empty((n_steps, n_envs),           np.int64)
    logp_buf    = np.empty((n_steps, n_envs),           np.float32)
    val_buf     = np.empty((n_steps, n_envs),           np.float32)
    aug_rew_buf = np.empty((n_steps, n_envs),           np.float32)
    env_rew_buf = np.empty((n_steps, n_envs),           np.float32)
    kl_buf      = np.empty((n_steps, n_envs),           np.float32)
    done_buf    = np.empty((n_steps, n_envs),           np.float32)

    ep_env_rew = np.zeros(n_envs, np.float32)
    ep_kl      = np.zeros(n_envs, np.float32)
    ep_steps   = np.zeros(n_envs, np.int32)
    completed_env_rew  = []
    completed_kl_bonus = []

    student_ac.eval()
    teacher.eval()
    obs = venv.obs.copy()

    for t in range(n_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

        # Teacher log-probs (frozen)
        teacher_logits, _   = teacher(obs_t)
        teacher_log_probs   = F.log_softmax(teacher_logits, dim=-1)  # [n_envs, n_actions]

        # Student action + value
        logits, vals = student_ac(obs_t)
        dist         = Categorical(logits=logits)
        acts         = dist.sample()
        logps        = dist.log_prob(acts)

        # KL bonus: log p_teacher(a_student | s)  — shape [n_envs]
        kl_bonus = teacher_log_probs.gather(1, acts.unsqueeze(1)).squeeze(1)

        obs_buf[t]     = obs
        act_buf[t]     = acts.cpu().numpy()
        logp_buf[t]    = logps.cpu().numpy()
        val_buf[t]     = vals.cpu().numpy()
        kl_buf[t]      = kl_bonus.cpu().numpy()

        obs, rews, dones, infos = venv.step(acts.cpu().numpy())

        env_rew_buf[t] = rews
        aug_rew_buf[t] = rews + kl_coef * kl_bonus.cpu().numpy()
        done_buf[t]    = dones

        ep_env_rew += np.array([info.get("apple_reward", 0) for info in infos])
        ep_kl      += kl_buf[t]
        ep_steps   += 1
        for i in range(n_envs):
            if dones[i]:
                completed_env_rew.append(ep_env_rew[i])
                completed_kl_bonus.append(ep_kl[i] / max(ep_steps[i], 1))
                ep_env_rew[i] = 0.0
                ep_kl[i]      = 0.0
                ep_steps[i]   = 0

    # Bootstrap value for GAE
    _, last_vals = student_ac(torch.tensor(venv.obs, dtype=torch.float32, device=device))
    last_vals = last_vals.cpu().numpy()

    # GAE on augmented rewards
    advantages = np.zeros((n_steps, n_envs), np.float32)
    gae        = np.zeros(n_envs, np.float32)
    for t in reversed(range(n_steps)):
        next_val = val_buf[t + 1] if t < n_steps - 1 else last_vals
        not_done = 1.0 - done_buf[t]
        delta    = aug_rew_buf[t] + gamma * next_val * not_done - val_buf[t]
        gae      = delta + gamma * gae_lambda * not_done * gae
        advantages[t] = gae

    returns = advantages + val_buf

    obs_f  = obs_buf.reshape(-1, obs_size)
    act_f  = act_buf.reshape(-1)
    logp_f = logp_buf.reshape(-1)
    adv_f  = advantages.reshape(-1)
    ret_f  = returns.reshape(-1)

    return obs_f, act_f, logp_f, adv_f, ret_f, completed_env_rew, completed_kl_bonus


# ── PPO update ────────────────────────────────────────────────────────────────
def ppo_update(
    model, optimizer,
    obs_np, acts_np, old_logps_np, advs_np, rets_np,
    clip_eps, value_coef, entropy_coef,
    n_epochs, mini_batch_size, max_grad_norm, device,
):
    advs_np = (advs_np - advs_np.mean()) / (advs_np.std() + 1e-8)

    obs_t  = torch.tensor(obs_np,       dtype=torch.float32, device=device)
    acts_t = torch.tensor(acts_np,      dtype=torch.long,    device=device)
    olp_t  = torch.tensor(old_logps_np, dtype=torch.float32, device=device)
    adv_t  = torch.tensor(advs_np,      dtype=torch.float32, device=device)
    ret_t  = torch.tensor(rets_np,      dtype=torch.float32, device=device)

    n = len(obs_np)
    total_loss = 0.0
    n_mb = 0

    model.train()
    for _ in range(n_epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, mini_batch_size):
            idx = perm[start: start + mini_batch_size]

            logits, vals = model(obs_t[idx])
            dist         = Categorical(logits=logits)
            new_logps    = dist.log_prob(acts_t[idx])
            entropy      = dist.entropy().mean()

            ratio       = (new_logps - olp_t[idx]).exp()
            adv_mb      = adv_t[idx]
            surr1       = ratio * adv_mb
            surr2       = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss  = F.mse_loss(vals, ret_t[idx])
            loss        = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            n_mb += 1

    return total_loss / n_mb


# ── eval ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_student(model, env, device, n_ep: int = 20) -> float:
    """Evaluate student (apple reward only, no KL bonus)."""
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
def train_student_rl(
    n_updates:       int   = 500,
    n_envs:          int   = 16,
    n_steps:         int   = 128,
    n_epochs:        int   = 4,
    mini_batch_size: int   = 256,
    gamma:           float = 0.99,
    gae_lambda:      float = 0.95,
    lr:              float = 1e-3,
    clip_eps:        float = 0.2,
    entropy_coef:    float = 0.01,
    value_coef:      float = 0.5,
    max_grad_norm:   float = 0.5,
    kl_coef:         float = 0.1,   # weight on log p_teacher(a|s) bonus
    eval_every:      int   = 20,    # evaluate student every N PPO updates
    teacher_path:    str   = "teacher.pt",
    output_path:     str   = "student_rl.pt",
    log_callback           = None,  # callable(env_steps, student_rew, avg_kl_bonus, entropy)
):
    device     = get_device()
    batch_size = n_steps * n_envs

    env  = AppleGridEnv()
    venv = VecEnvStudent(n_envs)

    # ── load teacher ──────────────────────────────────────────────────────────
    teacher = TeacherNet(env.obs_size, env.n_actions).to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
    teacher.eval()

    # ── build student actor-critic ────────────────────────────────────────────
    student_ac = StudentActorCritic(env.obs_size, env.n_actions).to(device)
    opt        = optim.Adam(student_ac.parameters(), lr=lr)

    n_teacher  = sum(p.numel() for p in teacher.parameters())
    n_student  = sum(p.numel() for p in StudentNet(env.obs_size, env.n_actions).parameters())
    print(f"Device         : {device}")
    print(f"Teacher params : {n_teacher:,}")
    print(f"Student params : {n_student:,}  ({100*n_student/n_teacher:.1f}% of teacher)")
    print(f"Batch size     : {n_steps} steps × {n_envs} envs = {batch_size:,} transitions")
    print(f"PPO updates    : {n_updates}  ({n_epochs} epochs each)")
    print(f"KL coef        : {kl_coef}  (log p_teacher bonus weight)\n")

    teacher_rew = evaluate_student(teacher, env, device, n_ep=50)
    print(f"Teacher baseline (50 ep stochastic): {teacher_rew:.2f}\n")

    header = (
        f"{'Upd':>5} | {'Steps':>9} | {'StudRew':>8} | "
        f"{'KLbonus':>8} | {'Elapsed':>8}"
    )
    print(header)
    print("─" * len(header))

    import collections
    env_rew_buf  = collections.deque(maxlen=200)
    kl_bonus_buf = collections.deque(maxlen=200)
    total_steps  = 0
    total_ep     = 0
    t0           = time.time()

    for upd in range(1, n_updates + 1):
        obs_f, act_f, logp_f, adv_f, ret_f, comp_env, comp_kl = collect_rollout(
            student_ac, teacher, venv, n_steps, kl_coef, device, gamma, gae_lambda
        )
        total_steps += batch_size
        total_ep    += len(comp_env)
        env_rew_buf.extend(comp_env)
        kl_bonus_buf.extend(comp_kl)

        # quick entropy estimate
        with torch.no_grad():
            logits_s, _ = student_ac(
                torch.tensor(obs_f[:min(512, len(obs_f))], dtype=torch.float32, device=device)
            )
            ent = Categorical(logits=logits_s).entropy().mean().item()

        ppo_update(
            student_ac, opt,
            obs_f, act_f, logp_f, adv_f, ret_f,
            clip_eps, value_coef, entropy_coef,
            n_epochs, mini_batch_size, max_grad_norm, device,
        )

        if upd % eval_every == 0:
            student_rew  = evaluate_student(student_ac, env, device, n_ep=20)
            avg_env_rew  = float(np.mean(env_rew_buf))  if env_rew_buf  else 0.0
            avg_kl_bonus = float(np.mean(kl_bonus_buf)) if kl_bonus_buf else 0.0
            elapsed      = time.time() - t0

            print(
                f"{upd:>5} | {total_steps:>9,} | {student_rew:>8.2f} | "
                f"{avg_kl_bonus:>8.3f} | {elapsed:>7.1f}s",
                flush=True,
            )

            if log_callback:
                log_callback(total_steps, student_rew, avg_kl_bonus, ent)
        else:
            elapsed = time.time() - t0
            avg_env_rew  = float(np.mean(env_rew_buf))  if env_rew_buf  else 0.0
            avg_kl_bonus = float(np.mean(kl_bonus_buf)) if kl_bonus_buf else 0.0
            print(
                f"{upd:>5} | {total_steps:>9,} | {'---':>8} | "
                f"{avg_kl_bonus:>8.3f} | {elapsed:>7.1f}s",
                flush=True,
            )

    # ── extract actor weights into compatible StudentNet ──────────────────────
    student_net = StudentNet(env.obs_size, env.n_actions)
    student_net.trunk.load_state_dict(student_ac.trunk.state_dict())
    student_net.actor.load_state_dict(student_ac.actor.state_dict())
    torch.save(student_net.state_dict(), output_path)
    print(f"\n✓  Saved {output_path}")

    # ── final head-to-head ────────────────────────────────────────────────────
    print("\n━━━  Final evaluation  ━━━  (50 episodes, stochastic policy)")
    t_rew = evaluate_student(teacher, env, device, n_ep=50)
    s_rew = evaluate_student(student_ac, env, device, n_ep=50)
    elapsed = time.time() - t0
    print(f"  Teacher : {t_rew:.2f}")
    print(f"  Student : {s_rew:.2f}  ({100*s_rew/t_rew:.1f}% of teacher reward)")
    print(f"  Params  : {n_teacher:,} → {n_student:,}  ({100*n_student/n_teacher:.1f}%)")
    print(f"  Total time : {elapsed:.1f}s")

    return student_ac


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--updates",   type=int,   default=500,
                   help="PPO update steps")
    p.add_argument("--envs",      type=int,   default=16)
    p.add_argument("--steps",     type=int,   default=128,
                   help="Rollout length per env")
    p.add_argument("--epochs",    type=int,   default=4,
                   help="PPO re-use epochs per rollout")
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--kl-coef",   type=float, default=0.1,
                   help="Weight on log p_teacher(a|s) reward bonus")
    p.add_argument("--teacher",   type=str,   default="teacher.pt")
    p.add_argument("--output",    type=str,   default="student_rl.pt")
    args = p.parse_args()

    train_student_rl(
        n_updates    = args.updates,
        n_envs       = args.envs,
        n_steps      = args.steps,
        n_epochs     = args.epochs,
        lr           = args.lr,
        kl_coef      = args.kl_coef,
        teacher_path = args.teacher,
        output_path  = args.output,
    )
