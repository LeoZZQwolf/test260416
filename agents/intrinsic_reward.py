"""
Self-Balancing Intrinsic Reward (Section IV-B, Eq. 15)

Helps UAVs balance between emergency (goal) and surveillance (anti-goal) tasks.

r_intr = -(1 - ω) * AoI_g_norm * d_g_norm + ω * d_ā_norm

- Goal g: the emergency PoI with highest priority
- Anti-goal ā: the nearest other UAV's location (to avoid duplication)
- ω ∈ [0, 1]: trade-off parameter
"""

import numpy as np
from typing import Optional, List
from env.uav_env import UAVState, PoI


class IntrinsicRewardComputer:
    """
    Computes self-balancing intrinsic reward for a UAV.

    Based on "sibling rivalry" concept from Trott et al., NeurIPS 2019.
    """

    def __init__(self, omega: float = 0.7, world_size: float = 6000.0):
        self.omega = omega
        self.world_size = world_size

    def compute(self, uav: UAVState, goal: Optional[PoI],
                other_uavs: List[UAVState]) -> float:
        """
        Compute intrinsic reward for a UAV.

        Args:
            uav: current UAV state
            goal: the emergency PoI assigned as goal (highest priority)
            other_uavs: list of other UAV states

        Returns:
            intrinsic reward value
        """
        if goal is None:
            return 0.0

        # Normalized distance to goal
        d_goal = np.sqrt((uav.x - goal.x)**2 + (uav.y - goal.y)**2)
        d_goal_norm = d_goal / self.world_size

        # Normalized AoI of goal
        aoi_goal_norm = goal.aoi / max(goal.aoi_threshold, 1)

        # Anti-goal: nearest other UAV (to avoid duplication of effort)
        min_dist_to_other = float('inf')
        for other in other_uavs:
            if other.uav_id != uav.uav_id:
                d = np.sqrt((uav.x - other.x)**2 + (uav.y - other.y)**2)
                min_dist_to_other = min(min_dist_to_other, d)

        if min_dist_to_other == float('inf'):
            min_dist_to_other = self.world_size

        d_antigoal_norm = min_dist_to_other / self.world_size

        # Self-balancing intrinsic reward (Eq. 15)
        r_intr = -(1 - self.omega) * aoi_goal_norm * d_goal_norm \
                 + self.omega * d_antigoal_norm

        return r_intr

    def compute_batch(self, uavs: List[UAVState],
                      goals: List[Optional[PoI]]) -> List[float]:
        """Compute intrinsic rewards for all UAVs."""
        rewards = []
        for i, uav in enumerate(uavs):
            r = self.compute(uav, goals[i], uavs)
            rewards.append(r)
        return rewards
