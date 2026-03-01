"""
Neural network definitions for the on-policy distillation demo.

TeacherNet  –  actor-critic, 2×256 hidden (shared trunk)  ~125k params
StudentNet  –  actor-only,   2×64  hidden                  ~19k  params
"""

import torch
import torch.nn as nn


class TeacherNet(nn.Module):
    """
    Larger actor-critic used for A2C training.
    forward(x) → (action_logits, state_value)
    """
    def __init__(self, obs_size: int, n_actions: int, hidden: int = 256):
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


class StudentNet(nn.Module):
    """
    Smaller actor-only network distilled from TeacherNet.
    forward(x) → action_logits
    """
    def __init__(self, obs_size: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden,   hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor):
        return self.actor(self.trunk(x))
