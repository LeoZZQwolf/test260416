"""
Dynamically Weighted Queue (Section IV-A)

Each UAV maintains a queue of assigned emergency PoIs.
The queue estimates priorities using the temporal predictor,
and the highest-priority PoI becomes the UAV's current goal.

This differs from PER: it manages task assignments during
both training and inference, not experience replay.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from env.uav_env import PoI, UAVState
from agents.temporal_predictor import TemporalPredictor


@dataclass
class QueueEntry:
    """An entry in the dynamically weighted queue."""
    poi: PoI
    priority: float  # lower = higher priority (estimated timeslots)


class DynamicallyWeightedQueue:
    """
    Queue of emergency PoIs for a single UAV.

    Maintains l_que entries, prioritized by temporal predictor.
    The entry with smallest estimated cost becomes the UAV's goal.
    """

    def __init__(self, uav_id: int, max_length: int = 3,
                 predictor: Optional[TemporalPredictor] = None):
        self.uav_id = uav_id
        self.max_length = max_length
        self.predictor = predictor
        self.entries: List[QueueEntry] = []

    def insert(self, poi: PoI, obs_features: np.ndarray):
        """
        Insert a new emergency PoI into the queue.

        If queue is full, remove the entry with the lowest priority
        (largest estimated cost).
        """
        # Compute initial priority
        priority = self._estimate_priority(obs_features, poi)

        entry = QueueEntry(poi=poi, priority=priority)
        self.entries.append(entry)

        # Trim to max length
        if len(self.entries) > self.max_length:
            self.entries.sort(key=lambda e: e.priority)
            self.entries = self.entries[:self.max_length]

    def update_priorities(self, uav_obs: np.ndarray, goal_features_fn):
        """
        Re-compute priorities for all entries using the temporal predictor.

        Called at each timeslot.
        """
        for entry in self.entries:
            if entry.poi.active:
                goal_feat = goal_features_fn(entry.poi)
                entry.priority = self._estimate_priority_with_features(
                    uav_obs, goal_feat
                )

    def get_top_goal(self) -> Optional[PoI]:
        """Get the emergency PoI with the highest priority (lowest cost)."""
        active_entries = [e for e in self.entries if e.poi.active]
        if not active_entries:
            return None
        active_entries.sort(key=lambda e: e.priority)
        return active_entries[0].poi

    def remove_handled(self):
        """Remove handled PoIs from the queue."""
        self.entries = [e for e in self.entries if e.poi.active]

    def _estimate_priority(self, obs: np.ndarray, poi: PoI) -> float:
        """
        Estimate priority (expected timeslots to handle).

        Uses temporal predictor if available, otherwise uses
        a simple heuristic: remaining AoI budget / distance ratio.
        """
        if self.predictor is not None:
            goal_feat = self._poi_to_features(poi)
            cost = self.predictor.predict_cost(obs, goal_feat)
            return cost
        else:
            # Heuristic fallback
            remaining = max(1, poi.aoi_threshold - poi.aoi)
            return float(remaining)

    def _estimate_priority_with_features(self, obs: np.ndarray,
                                          goal_feat: np.ndarray) -> float:
        if self.predictor is not None:
            return self.predictor.predict_cost(obs, goal_feat)
        return 1.0

    @staticmethod
    def _poi_to_features(poi: PoI) -> np.ndarray:
        """Convert PoI to feature vector."""
        return np.array([
            poi.x / 6000.0,
            poi.y / 6000.0,
            poi.aoi / max(poi.aoi_threshold, 1),
            (poi.aoi_threshold - poi.aoi) / max(poi.aoi_threshold, 1),
        ])

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        goals = [(e.poi.poi_id, f"{e.priority:.2f}") for e in self.entries]
        return f"Queue(UAV {self.uav_id}, len={len(self)}, goals={goals})"
