"""
Neural Network Architectures for DRL-MTUCS

- Policy network (actor): processes observation + goal → action
- Value network (critic): processes observation → value estimate
- High-level allocator network: processes global state → UAV assignment

Each UAV has its own policy θ_u and value network φ_u.
The allocator has its own policy θ_high and value φ_high.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class CNNFeatureExtractor(nn.Module):
    """
    CNN for processing AoI heatmap (Section IV-A).

    Extracts spatial features from the grid-based AoI map.
    """

    def __init__(self, grid_size: int = 16, output_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(32 * 4 * 4, output_dim)

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        Args:
            heatmap: (B, grid_size, grid_size) AoI heatmap
        Returns:
            (B, output_dim) spatial features
        """
        x = heatmap.unsqueeze(1)  # (B, 1, H, W)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class UAVPolicyNetwork(nn.Module):
    """
    Low-level policy network for a single UAV (Section IV-A).

    Processes: observation features + goal features → action (dx, dy, speed)

    Architecture: MLP layers with 128 hidden units (as per paper Table III).
    """

    def __init__(self, obs_dim: int = 32, goal_dim: int = 8,
                 action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Mean and log_std for continuous actions
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor,
                goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, log_std) of action distribution."""
        x = torch.cat([obs, goal], dim=-1)
        features = self.net(x)
        mean = torch.tanh(self.mean_head(features))  # [-1, 1] for dx, dy
        log_std = self.log_std_head(features).clamp(-5, 2)
        return mean, log_std

    def get_action(self, obs: torch.Tensor, goal: torch.Tensor,
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return log_prob."""
        mean, log_std = self.forward(obs, goal)
        std = log_std.exp()

        if deterministic:
            action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()

        # Ensure speed component is positive
        action_clamped = torch.cat([
            action[:, :2],  # dx, dy (keep tanh range)
            torch.sigmoid(action[:, 2:3]),  # speed in [0, 1]
        ], dim=-1)

        log_prob = torch.distributions.Normal(mean, std).log_prob(action).sum(-1)
        return action_clamped, log_prob

    def evaluate_action(self, obs: torch.Tensor, goal: torch.Tensor,
                        action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log_prob and entropy for given action."""
        mean, log_std = self.forward(obs, goal)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class UAVValueNetwork(nn.Module):
    """
    Low-level value network for a single UAV (Section IV-A).

    Processes: observation → value estimate V(s; φ)
    """

    def __init__(self, obs_dim: int = 32, goal_dim: int = 8,
                 hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, goal], dim=-1)
        return self.net(x)


class HighLevelAllocatorNetwork(nn.Module):
    """
    High-level policy network for UAV goal assignment (Section IV-A).

    Processes global state → assignment of emergency PoI to a UAV.

    State includes: emergency PoI locations, UAV local observations.
    Action: which UAV to assign (discrete, |U| options).
    """

    def __init__(self, state_dim: int = 64, num_uavs: int = 4,
                 hidden_dim: int = 128):
        super().__init__()
        self.num_uavs = num_uavs
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_uavs),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns action logits for each UAV."""
        return self.net(state)

    def get_action(self, state: torch.Tensor,
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample UAV assignment and return log_prob."""
        logits = self.forward(state)
        if deterministic:
            action = logits.argmax(dim=-1)
            log_prob = torch.zeros(state.size(0))
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_action(self, state: torch.Tensor,
                        action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


class HighLevelValueNetwork(nn.Module):
    """High-level value network for the allocator."""

    def __init__(self, state_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class FeatureEncoder:
    """
    Encodes raw observations into fixed-size feature vectors
    for use in the neural networks.
    """

    def __init__(self, grid_resolution: int = 16, max_emer: int = 10,
                 num_uavs: int = 4):
        self.grid_resolution = grid_resolution
        self.max_emer = max_emer
        self.num_uavs = num_uavs

    def encode_uav_obs(self, obs: dict) -> np.ndarray:
        """
        Encode a single UAV's observation into a flat vector.

        Components:
        - position (2,)
        - energy (1,)
        - speed (1,)
        - rel_uav_positions ((U-1)*2,)
        - AoI heatmap flattened (grid_res^2,)
        - emer_features (max_emer * 3,)
        """
        parts = [
            obs['position'],
            obs['energy'],
            obs['speed'],
            obs['rel_uav_positions'],
            obs['aoi_heatmap'].flatten(),
        ]
        # Pad emergency features to fixed size
        emer = obs['emer_features']
        if len(emer) < self.max_emer * 3:
            emer = np.pad(emer, (0, self.max_emer * 3 - len(emer)))
        else:
            emer = emer[:self.max_emer * 3]
        parts.append(emer)

        return np.concatenate(parts).astype(np.float32)

    def encode_goal(self, poi_features: np.ndarray) -> np.ndarray:
        """Encode goal PoI features into a fixed-size vector."""
        if len(poi_features) < 8:
            poi_features = np.pad(poi_features, (0, 8 - len(poi_features)))
        else:
            poi_features = poi_features[:8]
        return poi_features.astype(np.float32)

    def encode_global_state(self, global_obs: dict) -> np.ndarray:
        """Encode global state for high-level allocator."""
        parts = [
            global_obs['timeslot'],
            global_obs['uav_positions'],
            global_obs['uav_energies'],
        ]
        emer = global_obs['emer_positions']
        if isinstance(emer, np.ndarray) and len(emer) > 1:
            if len(emer) < self.max_emer * 3:
                emer = np.pad(emer, (0, self.max_emer * 3 - len(emer)))
            else:
                emer = emer[:self.max_emer * 3]
        parts.append(emer)
        return np.concatenate(parts).astype(np.float32)

    def poi_to_goal_features(self, poi) -> np.ndarray:
        """Convert a PoI to goal features (8-dim)."""
        return np.array([
            poi.x / 6000.0,
            poi.y / 6000.0,
            poi.aoi / max(poi.aoi_threshold, 1),
            (poi.aoi_threshold - poi.aoi) / max(poi.aoi_threshold, 1),
            float(poi.is_emergency),
            float(poi.active),
            poi.arrival_time / 120.0,
            0.0,  # padding
        ], dtype=np.float32)

    @property
    def uav_obs_dim(self) -> int:
        return 4 + (self.num_uavs - 1) * 2 + self.grid_resolution**2 + self.max_emer * 3

    @property
    def goal_dim(self) -> int:
        return 8

    @property
    def global_state_dim(self) -> int:
        return 1 + self.num_uavs * 2 + self.num_uavs + self.max_emer * 3
