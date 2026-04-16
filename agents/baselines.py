"""
Baseline algorithms for comparison (Section V-D).

1. Random: random actions
2. Greedy: always go to nearest/highest-AoI PoI
3. mTSP: multi-traveling salesman problem heuristic
"""

import numpy as np
from typing import List, Tuple, Optional
from env.uav_env import UAVCrowdsensingEnv, UAVState, PoI


class RandomBaseline:
    """Random policy baseline."""

    def __init__(self, num_uavs: int, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.num_uavs = num_uavs

    def act(self, obs: dict, env: UAVCrowdsensingEnv) -> List[Tuple[float, float, float]]:
        actions = []
        for i in range(self.num_uavs):
            dx = self.rng.uniform(-1, 1)
            dy = self.rng.uniform(-1, 1)
            speed = self.rng.uniform(0.3, 1.0)
            actions.append((dx, dy, speed))
        return actions


class GreedyBaseline:
    """
    Greedy policy: each UAV heads to the PoI with the highest AoI
    (or nearest emergency PoI).
    """

    def __init__(self, num_uavs: int):
        self.num_uavs = num_uavs

    def act(self, obs: dict, env: UAVCrowdsensingEnv) -> List[Tuple[float, float, float]]:
        actions = []
        used_pois = set()

        for i in range(self.num_uavs):
            if i not in obs:
                actions.append((0, 0, 0.5))
                continue

            uav = env.uavs[i]

            # Priority 1: active emergency PoIs
            best_poi = None
            best_score = -np.inf

            for poi in env.emer_pois:
                if not poi.active or poi.poi_id in used_pois:
                    continue
                dist = np.sqrt((uav.x - poi.x)**2 + (uav.y - poi.y)**2)
                urgency = poi.aoi / max(poi.aoi_threshold, 1)
                score = urgency / (dist + 1)
                if score > best_score:
                    best_score = score
                    best_poi = poi

            # Priority 2: high-AoI surveillance PoIs
            if best_poi is None:
                for poi in env.surv_pois:
                    if poi.poi_id in used_pois:
                        continue
                    dist = np.sqrt((uav.x - poi.x)**2 + (uav.y - poi.y)**2)
                    score = poi.aoi / (dist + 1)
                    if score > best_score:
                        best_score = score
                        best_poi = poi

            if best_poi is not None:
                used_pois.add(best_poi.poi_id)
                dx = best_poi.x - uav.x
                dy = best_poi.y - uav.y
                mag = np.sqrt(dx**2 + dy**2)
                if mag > 0:
                    dx /= mag
                    dy /= mag
                # Slow down near emergencies
                if best_poi.is_emergency:
                    speed = 0.3
                else:
                    speed = 0.8
                actions.append((dx, dy, speed))
            else:
                actions.append((0, 0, 0.5))

        return actions


class mTSPBaseline:
    """
    Simplified mTSP heuristic: plan routes for all UAVs to cover
    all PoIs, prioritizing emergencies.

    Based on: Shao & Xu, IEEE TVT 2023.
    """

    def __init__(self, num_uavs: int):
        self.num_uavs = num_uavs
        self.routes: List[List[PoI]] = [[] for _ in range(num_uavs)]

    def act(self, obs: dict, env: UAVCrowdsensingEnv) -> List[Tuple[float, float, float]]:
        # Re-plan routes periodically
        if env.current_timeslot % 10 == 0 or not any(self.routes):
            self._plan_routes(env)

        actions = []
        for i in range(self.num_uavs):
            if i not in obs:
                actions.append((0, 0, 0.5))
                continue

            uav = env.uavs[i]

            if self.routes[i]:
                target = self.routes[i][0]
                dist = np.sqrt((uav.x - target.x)**2 + (uav.y - target.y)**2)

                # Check if reached
                if dist < 100:
                    self.routes[i].pop(0)
                    if self.routes[i]:
                        target = self.routes[i][0]
                        dist = np.sqrt((uav.x - target.x)**2 + (uav.y - target.y)**2)
                    else:
                        actions.append((0, 0, 0.5))
                        continue

                dx = target.x - uav.x
                dy = target.y - uav.y
                mag = np.sqrt(dx**2 + dy**2)
                if mag > 0:
                    dx /= mag
                    dy /= mag

                speed = 0.3 if target.is_emergency else 0.7
                actions.append((dx, dy, speed))
            else:
                actions.append((0, 0, 0.5))

        return actions

    def _plan_routes(self, env: UAVCrowdsensingEnv):
        """Plan routes using nearest-neighbor heuristic."""
        all_pois = []
        # Emergencies first
        for p in env.emer_pois:
            if p.active:
                all_pois.append(p)
        # Then high-AoI surveillance
        surv_sorted = sorted(env.surv_pois, key=lambda p: -p.aoi)
        all_pois.extend(surv_sorted[:50])

        # Assign to nearest UAV (round-robin for simplicity)
        self.routes = [[] for _ in range(self.num_uavs)]
        for poi in all_pois:
            best_uav = 0
            best_dist = np.inf
            for i in range(self.num_uavs):
                uav = env.uavs[i]
                d = np.sqrt((uav.x - poi.x)**2 + (uav.y - poi.y)**2)
                if d < best_dist:
                    best_dist = d
                    best_uav = i
            self.routes[best_uav].append(poi)
