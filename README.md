# On-Policy Distillation Demo — A2C vs PPO

A self-contained RL + knowledge-distillation demo that trains **two teacher
algorithms** (A2C and PPO) on a 15×15 apple-collecting grid world, then
distils each teacher into an identical small student network via on-policy KL
minimisation.  A final comparison script evaluates all four models side-by-side.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Pipeline                                                                   │
│                                                                             │
│  AppleGridEnv  ──►  TeacherNet-A2C ──►  StudentNet-A2C                     │
│    15×15 grid   └►  TeacherNet-PPO ──►  StudentNet-PPO                     │
│    40-step ep          2×256, ~127k          2×64, ~19k                     │
│                       actor-critic           actor-only                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Repository layout

```
.
├── grid_env.py            # Environment: AppleGridEnv
├── models.py              # Neural networks: TeacherNet, StudentNet
├── train_teacher.py       # A2C teacher training
├── train_teacher_ppo.py   # PPO teacher training
├── train_student.py       # On-policy KL distillation (teacher → student)
├── visualize_student.py   # Animated episode viewer for any student
├── compare.py             # Evaluate & compare all four models
├── run_teacher.sh         # Launch: A2C teacher only (saves teacher.pt)
├── run_student.sh         # Launch: distil from teacher.pt → student.pt
├── run_experiment.sh      # Launch: A2C teacher → student → visualise
├── run_compare.sh         # Launch: full comparison (A2C + PPO, 4 models)
├── teacher_a2c.pt         # A2C teacher weights  (produced by run_compare.sh)
├── teacher_ppo.pt         # PPO teacher weights  (produced by run_compare.sh)
├── student_a2c.pt         # A2C-distilled student weights
├── student_ppo.pt         # PPO-distilled student weights
├── teacher.pt             # A2C teacher weights  (produced by run_teacher.sh)
└── student.pt             # Student weights      (produced by run_student.sh)
```

---

## File reference

### `grid_env.py` — AppleGridEnv

A pure-NumPy 15×15 grid world.

**State space** — 233-dimensional flat vector:

| Slice | Size | Contents |
|-------|------|----------|
| `[0:225]` | 225 | Flattened grid. `+1.0` = apple, `-1.0` = agent cell, `0` = empty |
| `[225:227]` | 2 | Agent row/col normalised to `[0, 1]` |
| `[227:233]` | 6 | Relative (Δrow, Δcol) to the 3 nearest apples, normalised to `[-1, 1]`. All zeros when fewer than 3 apples exist. |

**Action space** — discrete 4: `0`=up `1`=down `2`=left `3`=right.

**Reward** — `+1` for each apple collected (sparse). A potential-based shaping
bonus is added during all teacher training and always zero at evaluation.

**Dynamics:**
- Agent starts in the centre cell `(7, 7)`.
- 10 apples are placed on random empty cells at reset.
- Each step, with probability 0.10, the intended action is replaced by a
  uniformly random one (environment stochasticity).
- After the agent moves, a new apple spawns on a random empty cell with
  probability 0.10 per step.
- Episode ends after 40 steps.

**Shaping reward** (used only during teacher training):

```
shaping = (dist_before − dist_after) / max_dist × shaping_coef
```

where `dist_*` is the Manhattan distance to the nearest apple before/after
moving, `max_dist = 2(grid_size − 1)`, and `shaping_coef` is annealed from
`0.05 → 0` over training.  This is a standard potential-based bonus that is
theoretically guaranteed not to alter the optimal policy (Ng et al., 1999).

---

### `models.py` — TeacherNet / StudentNet

```
TeacherNet   obs_size → 256 → 256 → {actor head (4), critic head (1)}
             ~127 k parameters   actor-critic (used by A2C and PPO)

StudentNet   obs_size →  64 →  64 → actor head (4)
              ~19 k parameters   actor-only  (distillation target)
```

Both use ReLU activations throughout.  Both teacher algorithms share the same
`TeacherNet` architecture — this isolates the algorithm difference from any
capacity difference.

---

### `train_teacher.py` — A2C with potential-based shaping

Trains `TeacherNet` with the Advantage Actor-Critic (A2C) algorithm.

**Algorithm (one update):**
1. Reset all `n_envs` parallel environments.
2. Run each environment to completion under the current policy, collecting
   `(log_prob, value, entropy, reward)` tuples at every step.
3. Compute Monte-Carlo returns `G_t = Σ_{k≥t} γ^{k−t} r_k` (no bootstrapping).
4. Compute per-step advantage `A_t = G_t − V(s_t)`, then normalise to
   zero mean / unit std within each episode.
5. Losses:
   ```
   actor_loss  = −mean( log π(a|s) · A_t )
   critic_loss = MSE( V(s), G )
   total       = actor_loss + 0.5 · critic_loss
   ```
6. Clip gradient norm to `0.5`, Adam step.

**Hyperparameters (defaults):**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `n_updates` | 4 000 | Total gradient steps |
| `n_envs` | 16 | Parallel rollouts per update |
| `gamma` | 0.99 | Discount factor |
| `lr` | 3e-4 | Adam learning rate |
| `value_coef` | 0.5 | Weight on critic loss |
| `entropy_coef` | 0.0 | Entropy bonus (off by default) |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `shaping_init` | 0.05 | Initial shaping coefficient |
| `anneal_frac` | 0.75 | Fraction of updates over which shaping → 0 |

**Total env steps (default):** 4 000 updates × 16 envs × 40 steps ≈ **2.56 M**.

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--updates N` | 4000 | Number of A2C gradient steps |
| `--envs N` | 16 | Parallel environments |
| `--lr LR` | 3e-4 | Adam learning rate |
| `--no-wandb` | off | Disable W&B logging |
| `--no-plot` | off | Disable live Matplotlib window |
| `--output PATH` | `teacher.pt` | Where to save weights |

**Live plot** — 3-panel dark-theme dashboard: rolling apple reward vs
random/oracle baselines · policy entropy · critic loss + shaping coef.

---

### `train_teacher_ppo.py` — PPO with GAE

Trains `TeacherNet` with Proximal Policy Optimisation.

**Algorithm (one update):**
1. Collect exactly `n_steps` steps from all `n_envs` environments simultaneously
   (persistent state — envs auto-reset on episode end, episode boundaries
   recorded via `done=1` flags).
   Total batch = `n_steps × n_envs` transitions.
2. Compute Generalised Advantage Estimates (GAE) in a reverse scan:
   ```
   not_done  = 1 − done[t]
   δ_t       = r_t + γ · V(s_{t+1}) · not_done − V(s_t)
   Â_t       = δ_t + γλ · not_done · Â_{t+1}
   returns_t = Â_t + V(s_t)
   ```
   `not_done` masks out bootstrap and resets advantage accumulation at every
   episode boundary.  Bootstrap value for the final observation is computed
   with a separate forward pass.
3. Normalise advantages over the full batch.
4. For `n_epochs` passes, shuffle the batch into mini-batches of size
   `mini_batch_size` and update with the clipped surrogate:
   ```
   ratio        = exp( log π_new(a|s) − log π_old(a|s) )
   policy_loss  = −mean( min( ratio · Â,  clip(ratio, 1±ε) · Â ) )
   value_loss   = MSE( V_new(s), returns )
   total        = policy_loss + 0.5 · value_loss − 0.01 · entropy
   ```
5. Clip gradient norm to `0.5`, Adam step.

**Hyperparameters (defaults):**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `total_timesteps` | 2 500 000 | Total env steps (≈ same as A2C default) |
| `n_envs` | 16 | Parallel environments |
| `n_steps` | 128 | Rollout length per env; batch = 2 048 |
| `n_epochs` | 4 | PPO re-use epochs per rollout |
| `mini_batch_size` | 256 | SGD mini-batch size (8 per epoch) |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE λ |
| `lr` | 3e-4 | Adam learning rate |
| `clip_eps` | 0.2 | PPO clip epsilon |
| `entropy_coef` | 0.01 | Entropy bonus (mild exploration) |
| `value_coef` | 0.5 | Weight on value loss |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `shaping_init` | 0.05 | Initial shaping coefficient |
| `anneal_frac` | 0.75 | Fraction of timesteps over which shaping → 0 |

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--timesteps N` | 2500000 | Total env steps |
| `--envs N` | 16 | Parallel environments |
| `--steps N` | 128 | Rollout length per env |
| `--epochs N` | 4 | PPO re-use epochs |
| `--lr LR` | 3e-4 | Adam learning rate |
| `--clip F` | 0.2 | Clip epsilon |
| `--no-wandb` | off | Disable W&B logging |
| `--no-plot` | off | Disable live Matplotlib window |
| `--output PATH` | `teacher_ppo.pt` | Where to save weights |

**Console output columns:** `Upd | Episodes | Apples | Entropy | ValLoss | ClipFrac | KL | ShpCoef | Time`

**Live plot** — same 3-panel layout as A2C: rolling reward · entropy · value
loss + clip fraction (instead of shaping coef on the secondary axis).

---

### `train_student.py` — On-policy KL distillation

Distils any `TeacherNet` into the smaller `StudentNet` via on-policy soft
cross-entropy minimisation.

**Why on-policy?**  The teacher's action distribution is only meaningful at
states the teacher actually visits.  Collecting rollouts under the teacher's
own policy ensures the student trains on the same distribution the teacher has
mastered, rather than on arbitrary states.

**Algorithm (one iteration):**
1. **Collect** — run all `n_envs` environments simultaneously to completion
   under the frozen teacher, recording `(obs, teacher_softmax_probs)`.
2. **Shuffle** all `(obs, probs)` pairs.
3. **Distil** — minimise soft cross-entropy in mini-batches:
   ```
   L = −Σ_a p_teacher(a|s) · log p_student(a|s)
         ≡ KL( p_teacher ‖ p_student ) + H( p_teacher )
   ```
4. **Evaluate** every `eval_every` iterations.

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--teacher PATH` | `teacher.pt` | Path to teacher weights |
| `--output PATH` | `student.pt` | Path to save student weights |

---

### `visualize_student.py` — Animated episode viewer

Loads any `student.pt`-compatible file and renders full episodes in a
dark-theme Matplotlib window.

**Layout:**

```
┌────────────────────────┬─────────────────────────┐
│                        │   Action probabilities  │
│   15×15 grid           │   (live bar chart)      │
│   green = agent        ├─────────────────────────┤
│   orange = apple       │   Cumulative reward     │
│                        │   (step curve + fill)   │
└────────────────────────┴─────────────────────────┘
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--n N` | 3 | Number of episodes to visualise |
| `--delay S` | 0.25 | Seconds of pause between steps |
| `--greedy` | off | Use argmax instead of stochastic sampling |

---

### `compare.py` — Model comparison

Evaluates all four trained models plus the random and oracle baselines, then
prints a formatted comparison table.  Gracefully skips any `.pt` file that is
missing, so you can run it after training just the A2C pair if desired.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Comparison  (200 stochastic episodes each)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model               Params     Apples/ep  ±std  vs oracle
  ──────────────────  ─────────  ─────────  ────  ─────────
  Random              —               1.3   0.9       16%
  Oracle (greedy)     —               8.2   1.1      100%
  Teacher A2C         126,981         X.X   X.X       XX%
  Teacher PPO         126,981         X.X   X.X       XX%
  Student A2C          19,396         X.X   X.X       XX%
  Student PPO          19,396         X.X   X.X       XX%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--teacher-a2c PATH` | `teacher_a2c.pt` | A2C teacher weights |
| `--teacher-ppo PATH` | `teacher_ppo.pt` | PPO teacher weights |
| `--student-a2c PATH` | `student_a2c.pt` | A2C student weights |
| `--student-ppo PATH` | `student_ppo.pt` | PPO student weights |
| `--episodes N` | 200 | Evaluation episodes per model |

---

### `run_teacher.sh` — A2C teacher only

Trains a single A2C teacher and saves `teacher.pt`.  Accepts the same flags
as `train_teacher.py` (forwarded verbatim).

```bash
bash run_teacher.sh                  # defaults
bash run_teacher.sh --updates 8000   # longer run
bash run_teacher.sh --no-plot --no-wandb
```

---

### `run_student.sh` — Student distillation only

Distils `teacher.pt` into `student.pt`.  Aborts if `teacher.pt` is missing.

```bash
bash run_student.sh
```

---

### `run_experiment.sh` — Single A2C pipeline

Chains A2C teacher → student distillation → 3-episode visualisation.

```bash
bash run_experiment.sh               # full run with defaults
bash run_experiment.sh --no-plot     # headless training, then open viz
bash run_experiment.sh --updates 2000 --no-wandb
```

---

### `run_compare.sh` — Full A2C vs PPO comparison

The main experiment script.  Runs all four stages and prints the comparison
table at the end.

```bash
bash run_compare.sh                            # full run (~35 min)
bash run_compare.sh --no-plot --no-wandb       # headless
# Quick smoke-test (a few minutes):
bash run_compare.sh --no-plot --no-wandb --updates 200 --timesteps 200000
```

**Flag routing:**

| Flag | Routed to |
|------|-----------|
| `--no-plot`, `--no-wandb` | Both teachers |
| `--updates N`, `--envs N`, `--lr LR` | A2C teacher only |
| `--timesteps N`, `--steps N`, `--epochs N`, `--clip F` | PPO teacher only |

---

## Setup

```bash
conda create -n try python=3.11 -y
conda activate try
pip install torch numpy matplotlib
# Optional — experiment tracking:
pip install wandb && wandb login
```

Python ≥ 3.9 required.  Device backend auto-detected: Apple MPS → CUDA → CPU.

---

## Quick start

### Compare A2C vs PPO (full experiment)

```bash
bash run_compare.sh --no-wandb
```

Produces `teacher_a2c.pt`, `teacher_ppo.pt`, `student_a2c.pt`, `student_ppo.pt`
and prints the comparison table.

### Single A2C pipeline

```bash
bash run_teacher.sh                           # ~15 min → teacher.pt
bash run_student.sh                           # ~1 min  → student.pt
conda run -n try python visualize_student.py  # watch 3 episodes
```

### Evaluate existing weights

```bash
conda run -n try python compare.py            # all four default filenames
conda run -n try python compare.py --teacher-a2c teacher.pt --student-a2c student.pt
```

### Visualise a specific student

```bash
conda run -n try python visualize_student.py            # stochastic
conda run -n try python visualize_student.py --greedy   # argmax
conda run -n try python visualize_student.py --n 10     # 10 episodes
```

---

## How it works

### Step 1 — Teacher A2C

Standard advantage actor-critic with Monte-Carlo returns (no n-step or GAE).
Sixteen parallel environments provide decorrelated rollouts; each update
processes one complete episode from every environment before taking a single
gradient step.

The **potential-based shaping reward** acts as a curriculum: in early training
the agent receives a small dense signal for reducing its Manhattan distance to
the nearest apple.  The coefficient is linearly annealed to zero at 75% of
training, so the final policy is evaluated and optimised on sparse apple rewards
only.  Potential-based shaping is policy-invariant by construction.

### Step 2 — Teacher PPO

PPO collects a fixed-length batch of `n_steps × n_envs = 2 048` transitions per
update, then re-uses each sample `n_epochs = 4` times via shuffled mini-batches.
Two mechanisms prevent destructive over-optimisation of the stale data:

- **Clipped surrogate** — the policy improvement step is bounded so the new
  policy stays within a `[1−ε, 1+ε]` probability-ratio envelope of the old one.
- **GAE** — advantage estimates blend Monte-Carlo and TD(0) targets via `λ=0.95`,
  reducing variance without discarding long-horizon credit assignment.

The mild entropy bonus (`0.01`) prevents premature policy collapse.  The same
potential-based shaping schedule as A2C is used, annealed over total timesteps
rather than gradient steps.

PPO's sample re-use and on-policy corrections typically yield **higher asymptotic
performance than A2C** at the same total env-step budget.

### Step 3 — On-policy KL distillation

Both teachers are distilled using the same procedure: the teacher policy is
frozen, its rollouts generate `(state, action_distribution)` pairs, and the
student minimises the soft cross-entropy (= KL divergence from teacher to
student) over those pairs.

Using the **teacher's own rollouts** (on-policy w.r.t. the teacher) is
important: states the teacher never visits carry no useful signal, and training
on them can introduce compounding errors if the student drifts into unfamiliar
regions.

### Typical results

| Policy | Parameters | Apples / episode |
|--------|------------|-----------------|
| Random | — | ~1.3 |
| Oracle (greedy) | — | ~8.2 |
| Teacher A2C | 126 981 | ~3.7 |
| Teacher PPO | 126 981 | ~5.0 |
| Student-A2C | 19 396 (15 %) | ~3.5 (≈ 95 % of A2C teacher) |
| Student-PPO | 19 396 (15 %) | ~4.8 (≈ 96 % of PPO teacher) |

PPO typically outperforms A2C at the same env-step budget thanks to better
sample efficiency from policy re-use, GAE variance reduction, and the entropy
regulariser.  Both students recover ~95–97 % of their respective teacher's
reward at 1/7th the parameter count.

---

## Extending the demo

| Idea | Where to change |
|------|----------------|
| Larger grid or more apples | `AppleGridEnv.__init__` defaults in `grid_env.py` |
| Deeper / wider networks | `hidden` argument in `TeacherNet` / `StudentNet` (`models.py`) |
| PPO with linear LR annealing | Add a schedule in `train_teacher_ppo.py` |
| Temperature annealing in distillation | Add a schedule to `temperature` in `train_student.py` |
| DAgger (student self-improvement) | After distillation, mix teacher and student rollouts |
| W&B hyperparameter sweep | `wandb sweep` pointing at either teacher CLI |
| Visualise PPO student | `python visualize_student.py` after pointing `student.pt` at `student_ppo.pt` |
