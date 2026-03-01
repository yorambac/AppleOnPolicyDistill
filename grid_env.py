"""
AppleGridEnv  –  15x15 grid world for RL / distillation demos.

Observation  : flat grid (225 floats, 0=empty 1=apple) + normalised agent pos (2 floats) = 227 dims
Actions      : 0=up  1=down  2=left  3=right
Reward       : +1 for stepping on an apple
Stochasticity: with prob `stochastic_prob` the intended action is replaced by a random one
Apples       : `n_apples_init` placed at reset; each step spawns a new one with prob `apple_spawn_prob`
Episode      : terminates after `episode_length` steps
"""

import numpy as np


class AppleGridEnv:
    DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right

    def __init__(
        self,
        grid_size: int = 15,
        episode_length: int = 40,
        n_apples_init: int = 10,
        apple_spawn_prob: float = 0.10,
        stochastic_prob: float = 0.10,
    ):
        self.grid_size       = grid_size
        self.episode_length  = episode_length
        self.n_apples_init   = n_apples_init
        self.apple_spawn_prob = apple_spawn_prob
        self.stochastic_prob = stochastic_prob
        self.n_actions       = 4
        # grid(225) + agent_pos(2) + rel_pos to 3 nearest apples(6) = 233
        self.obs_size        = grid_size * grid_size + 2 + 6
        # shaping scale — set externally to anneal from 0.05 → 0 during training
        self.shaping_coef    = 0.05

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.row  = self.grid_size // 2
        self.col  = self.grid_size // 2
        self.t    = 0
        for _ in range(self.n_apples_init):
            self._spawn_apple()
        return self._obs()

    def _nearest_apple_dist(self) -> float:
        """Manhattan distance to nearest apple, or 0 if none."""
        apples = np.argwhere(self.grid == 1.0)
        if len(apples) == 0:
            return 0.0
        return float(np.min(np.abs(apples[:, 0] - self.row) + np.abs(apples[:, 1] - self.col)))

    # ------------------------------------------------------------------
    def step(self, action: int):
        # stochastic slip
        if np.random.random() < self.stochastic_prob:
            action = np.random.randint(self.n_actions)

        # distance to nearest apple BEFORE moving (for shaping)
        dist_before = self._nearest_apple_dist()

        dr, dc   = self.DELTAS[action]
        self.row = int(np.clip(self.row + dr, 0, self.grid_size - 1))
        self.col = int(np.clip(self.col + dc, 0, self.grid_size - 1))

        # sparse apple reward
        apple_reward = float(self.grid[self.row, self.col])
        self.grid[self.row, self.col] = 0.0

        # dense shaping: reward for reducing distance to nearest apple.
        # Use potential-based shaping F = γ·Φ(s') - Φ(s) with Φ = -dist/max_dist.
        # This is theoretically guaranteed not to change the optimal policy.
        dist_after  = self._nearest_apple_dist()
        max_dist    = 2.0 * (self.grid_size - 1)   # max possible Manhattan dist
        shaping     = (dist_before - dist_after) / max_dist   # in [-1, +1]
        # scale controlled by self.shaping_coef (annealed from 0.05 → 0 during training)
        shaping    *= self.shaping_coef

        if np.random.random() < self.apple_spawn_prob:
            self._spawn_apple()

        self.t += 1
        reward = apple_reward + shaping
        return self._obs(), reward, self.t >= self.episode_length, {
            "apple_reward": apple_reward, "shaping": shaping
        }

    # ------------------------------------------------------------------
    def _spawn_apple(self):
        """Place an apple on a random empty cell that isn't the agent."""
        empties = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if self.grid[r, c] == 0.0 and not (r == self.row and c == self.col)
        ]
        if empties:
            r, c = empties[np.random.randint(len(empties))]
            self.grid[r, c] = 1.0

    def _obs(self) -> np.ndarray:
        # grid with agent marked as -1 (apple=+1, agent=-1, empty=0)
        view = self.grid.copy()
        view[self.row, self.col] = -1.0

        # normalised absolute position
        pos = np.array([self.row, self.col], dtype=np.float32) / (self.grid_size - 1)

        # relative positions to the 3 nearest apples (normalised to [-1, 1])
        # gives the MLP a direct spatial signal without needing to do grid search
        apple_rc = np.argwhere(self.grid == 1.0)
        scale    = float(self.grid_size - 1)
        near_feats = np.zeros(6, dtype=np.float32)
        if len(apple_rc) > 0:
            dists = np.abs(apple_rc[:, 0] - self.row) + np.abs(apple_rc[:, 1] - self.col)
            for k, idx in enumerate(np.argsort(dists)[:3]):
                near_feats[k*2]   = (apple_rc[idx, 0] - self.row) / scale
                near_feats[k*2+1] = (apple_rc[idx, 1] - self.col) / scale

        return np.concatenate([view.flatten(), pos, near_feats])

    # ------------------------------------------------------------------
    def render(self):
        lines = []
        for r in range(self.grid_size):
            row = []
            for c in range(self.grid_size):
                if r == self.row and c == self.col:
                    row.append("\033[92m@\033[0m")   # green agent
                elif self.grid[r, c] == 1.0:
                    row.append("\033[93mA\033[0m")   # yellow apple
                else:
                    row.append(".")
            lines.append(" ".join(row))
        print("\n".join(lines))
        print(f"  step={self.t}  apples_left={int(self.grid.sum())}\n")
