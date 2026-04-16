"""
Temporal Predictor (Section IV-A, Eq. 14)

Estimates the expected number of timeslots for a UAV to reach
a goal observation from its current observation.

Nπ(o_t, o_t'; ψ) = Eπ[cost(o_t, o_t')]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


class TemporalPredictor(nn.Module):
    """
    Neural network temporal predictor parameterized by ψ.

    Input: concatenated features of current state and goal state.
    Output: predicted timeslots to reach the goal.
    """

    def __init__(self, obs_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # ensure positive output
        )

    def forward(self, obs: torch.Tensor, goal_obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: current observation features (B, obs_dim)
            goal_obs: goal observation features (B, obs_dim)
        Returns:
            predicted timeslots (B, 1)
        """
        x = torch.cat([obs, goal_obs], dim=-1)
        return self.net(x)

    def predict_cost(self, obs: np.ndarray, goal_obs: np.ndarray) -> float:
        """Predict cost in numpy (inference mode)."""
        self.eval()
        with torch.no_grad():
            o = torch.FloatTensor(obs).unsqueeze(0)
            g = torch.FloatTensor(goal_obs).unsqueeze(0)
            return self(o, g).item()


class TemporalPredictorTrainer:
    """
    Trains the temporal predictor using supervised regression (Eq. 16).

    L(ψ) = 1/2 E[(Nπ(o_i, o_j; ψ) - (j - i))^2]
    """

    def __init__(self, predictor: TemporalPredictor, lr: float = 1e-3):
        self.predictor = predictor
        self.optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)

    def update(self, trajectories: List[List[np.ndarray]]) -> float:
        """
        Update predictor from trajectory segments.

        Each trajectory is a list of observations from start of emergency
        handling to completion. We form (o_i, o_j, j-i) training pairs.
        """
        if not trajectories:
            return 0.0

        pairs = []
        for traj in trajectories:
            n = len(traj)
            # Form pairs from start to each subsequent state
            for j in range(1, n):
                pairs.append((traj[0], traj[j], j))

        if not pairs:
            return 0.0

        obs_batch = torch.FloatTensor([p[0] for p in pairs])
        goal_batch = torch.FloatTensor([p[1] for p in pairs])
        target_batch = torch.FloatTensor([p[2] for p in pairs]).unsqueeze(-1)

        self.predictor.train()
        pred = self.predictor(obs_batch, goal_batch)
        loss = F.mse_loss(pred, target_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
