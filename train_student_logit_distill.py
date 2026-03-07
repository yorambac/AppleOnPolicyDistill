"""
Forward-KL logit distillation: train StudentNet to mimic TeacherNet.

This is the *forward* KL direction: KL( p_teacher ‖ p_student ).
The teacher's state distribution drives training — rollouts are collected
under the *teacher* policy and the student minimises surprise at what
the teacher does.

Algorithm (each iteration)
──────────────────────────
1. Roll out the *teacher* policy for `n_envs` full episodes simultaneously,
   collecting (observation, teacher_softmax_probs) pairs at every step.
   This is "on-policy w.r.t. the teacher" — state distribution = teacher's.

2. Update the student by minimising the forward KL divergence
       KL( p_teacher ‖ p_student )
     = Σ_a p_teacher(a|s) log[ p_teacher(a|s) / p_student(a|s) ]
     = CE( p_teacher, p_student ) − H( p_teacher )

   Since H(p_teacher) is constant w.r.t. student weights, this is
   equivalent to the soft cross-entropy loss:
       L = − Σ_a p_teacher(a|s) · log p_student(a|s)

3. Every `eval_every` iterations: evaluate student reward and log.

Contrast with train_student_rl_distill.py which uses the *reverse* KL
direction — KL( p_student ‖ p_teacher ) — via PPO on the student's own
rollouts with a teacher-approval reward bonus.

Speed: rollouts use a batched VecEnv so the GPU does one forward pass
       per step across all parallel envs.

Saves the final student to student.pt.
"""

import time
import numpy as np
import torch
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


# ── vectorised env ─────────────────────────────────────────────────────────────
class VecEnv:
    def __init__(self, n: int, **kw):
        self.envs = [AppleGridEnv(**kw) for _ in range(n)]
        self.n = n

    def reset(self):
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions):
        results = [e.step(int(a)) for e, a in zip(self.envs, actions)]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.array(rews, np.float32), np.array(dones), infos


# ── helpers ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_teacher_rollouts_batched(
    teacher: TeacherNet,
    venv: VecEnv,
    device,
    temperature: float = 1.0,
):
    """
    Run all venv.n environments simultaneously under teacher policy.
    Returns:
      obs_np   : (N_steps_total, obs_size)
      probs_np : (N_steps_total, n_actions)  – teacher soft targets
      mean_ret : float
    """
    teacher.eval()
    obs_batch = venv.reset()                 # (N, obs)
    active    = np.ones(venv.n, bool)
    ep_rets   = np.zeros(venv.n, np.float32)

    all_obs   = []
    all_probs = []

    while active.any():
        active_idx = np.where(active)[0]
        obs_t      = torch.tensor(obs_batch[active_idx], dtype=torch.float32, device=device)

        logits, _  = teacher(obs_t)
        probs      = F.softmax(logits / temperature, dim=-1)
        actions    = Categorical(probs=probs).sample()

        # store (obs, teacher_probs) for active envs
        for local_i, env_i in enumerate(active_idx):
            all_obs.append(obs_batch[env_i].copy())
            all_probs.append(probs[local_i].cpu().numpy())

        actions_np = actions.cpu().numpy()
        for local_i, env_i in enumerate(active_idx):
            obs_n, r, done, _ = venv.envs[env_i].step(int(actions_np[local_i]))
            obs_batch[env_i]  = obs_n
            ep_rets[env_i]   += r
            if done:
                active[env_i] = False

    return (
        np.array(all_obs,   dtype=np.float32),
        np.array(all_probs, dtype=np.float32),
        float(ep_rets.mean()),
    )


@torch.no_grad()
def evaluate_stochastic(model, env: AppleGridEnv, device, n_episodes: int = 20) -> float:
    """Evaluate with stochastic sampling (matches training distribution)."""
    total = 0.0
    for _ in range(n_episodes):
        obs  = env.reset()
        done = False
        while not done:
            x      = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            out    = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            action = Categorical(logits=logits).sample().item()
            obs, r, done, _ = env.step(action)
            total += r
    return total / n_episodes


# ── main ──────────────────────────────────────────────────────────────────────
def train_student(
    n_iterations:      int   = 200,
    n_envs:            int   = 16,    # parallel envs for teacher rollouts
    batch_size:        int   = 512,
    lr:                float = 1e-3,
    temperature:       float = 1.0,
    eval_every:        int   = 10,    # evaluate student every N iters
    teacher_path:      str   = "teacher.pt",
    output_path:       str   = "student.pt",
    log_callback             = None,  # callable(env_steps, kl_loss, teacher_rew, student_rew)
):
    device = get_device()
    print(f"Device : {device}  |  parallel envs : {n_envs}\n")

    env  = AppleGridEnv()
    venv = VecEnv(n_envs)

    # ── load teacher ─────────────────────────────────────────────────────────
    teacher = TeacherNet(env.obs_size, env.n_actions).to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
    teacher.eval()

    # ── build student ─────────────────────────────────────────────────────────
    student = StudentNet(env.obs_size, env.n_actions).to(device)
    opt     = optim.Adam(student.parameters(), lr=lr)

    n_teacher = sum(p.numel() for p in teacher.parameters())
    n_student = sum(p.numel() for p in student.parameters())
    print(f"Teacher params : {n_teacher:,}")
    print(f"Student params : {n_student:,}  ({100*n_student/n_teacher:.1f}% of teacher)\n")

    teacher_base = evaluate_stochastic(teacher, env, device, n_episodes=50)
    print(f"Teacher baseline (50 ep stochastic): {teacher_base:.2f}\n")

    header = (
        f"{'Iter':>5} | {'KL loss':>8} | {'TeachRew':>9} | "
        f"{'StudRew':>8} | {'Gap':>7} | {'Elapsed':>8}"
    )
    print(header)
    print("─" * len(header))

    t0 = time.time()

    for it in range(1, n_iterations + 1):
        # ── 1. collect on-policy teacher data (batched) ───────────────────────
        obs_np, probs_np, teacher_rew = collect_teacher_rollouts_batched(
            teacher, venv, device, temperature
        )

        # shuffle
        perm     = np.random.permutation(len(obs_np))
        obs_np   = obs_np[perm]
        probs_np = probs_np[perm]

        # ── 2. distil: minimise KL(teacher ‖ student) ────────────────────────
        student.train()
        total_loss, n_batches = 0.0, 0

        for start in range(0, len(obs_np), batch_size):
            obs_t    = torch.tensor(obs_np[start: start + batch_size],
                                    dtype=torch.float32, device=device)
            target_t = torch.tensor(probs_np[start: start + batch_size],
                                    dtype=torch.float32, device=device)

            student_log_probs = F.log_softmax(student(obs_t) / temperature, dim=-1)
            loss = -(target_t * student_log_probs).sum(dim=-1).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches  += 1

        # ── 3. log ────────────────────────────────────────────────────────────
        env_steps = it * n_envs * env.episode_length   # approx teacher steps
        if it % eval_every == 0:
            student_rew = evaluate_stochastic(student, env, device, n_episodes=20)
            gap         = teacher_rew - student_rew
            elapsed     = time.time() - t0
            print(
                f"{it:>5} | {total_loss/n_batches:>8.4f} | "
                f"{teacher_rew:>9.2f} | {student_rew:>8.2f} | "
                f"{gap:>+7.2f} | {elapsed:>7.1f}s"
            )
            if log_callback:
                log_callback(env_steps, total_loss / n_batches, teacher_rew, student_rew)
        else:
            # lightweight per-iter print (no eval)
            elapsed = time.time() - t0
            print(
                f"{it:>5} | {total_loss/n_batches:>8.4f} | "
                f"{teacher_rew:>9.2f} | {'---':>8} | {'---':>7} | {elapsed:>7.1f}s",
                flush=True,
            )

    torch.save(student.state_dict(), output_path)
    print(f"\n✓  Saved {output_path}")

    # ── final head-to-head ────────────────────────────────────────────────────
    print("\n━━━  Final evaluation  ━━━  (50 episodes, stochastic policy)")
    t_rew = evaluate_stochastic(teacher, env, device, 50)
    s_rew = evaluate_stochastic(student, env, device, 50)
    elapsed = time.time() - t0
    print(f"  Teacher : {t_rew:.2f}")
    print(f"  Student : {s_rew:.2f}  ({100*s_rew/t_rew:.1f}% of teacher reward)")
    print(f"  Params  : {n_teacher:,} → {n_student:,}  ({100*n_student/n_teacher:.1f}%)")
    print(f"  Total time : {elapsed:.1f}s")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", type=str, default="teacher.pt",
                   help="Path to teacher weights file")
    p.add_argument("--output",  type=str, default="student.pt",
                   help="Path to save student weights")
    args = p.parse_args()
    train_student(teacher_path=args.teacher, output_path=args.output)
