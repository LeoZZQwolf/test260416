"""
DRL-MTUCS Main Agent (Algorithm 1)

Orchestrates:
1. High-level allocator: assigns emergency PoIs to UAVs (when new emergencies arrive)
2. Dynamically weighted queues: prioritize emergency PoIs per UAV
3. Low-level UAV execution: continuous action (dx, dy, speed)
4. Self-balancing intrinsic reward: goal vs anti-goal
5. Training: update all parameters after each episode
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple

from env.uav_env import UAVCrowdsensingEnv, SimConfig, PoI, UAVState
from agents.networks import (
    UAVPolicyNetwork, UAVValueNetwork,
    HighLevelAllocatorNetwork, HighLevelValueNetwork,
    FeatureEncoder
)
from agents.weighted_queue import DynamicallyWeightedQueue
from agents.intrinsic_reward import IntrinsicRewardComputer
from agents.temporal_predictor import TemporalPredictor, TemporalPredictorTrainer
from agents.ppo_trainer import PPOTrainer, RolloutBuffer


class DRLMTUCS:
    """
    DRL-MTUCS: Multi-Task-Oriented Emergency-Aware UAV Crowdsensing.

    Implements Algorithm 1 from the paper.
    """

    def __init__(self, config: Optional[SimConfig] = None):
        self.cfg = config or SimConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Feature encoder
        self.encoder = FeatureEncoder(
            grid_resolution=self.cfg.grid_resolution,
            num_uavs=self.cfg.num_uavs,
        )

        # High-level allocator networks
        self.alloc_policy = HighLevelAllocatorNetwork(
            state_dim=self.encoder.global_state_dim,
            num_uavs=self.cfg.num_uavs,
            hidden_dim=128
        ).to(self.device)
        self.alloc_value = HighLevelValueNetwork(
            state_dim=self.encoder.global_state_dim,
            hidden_dim=128
        ).to(self.device)

        # Low-level UAV networks (independent per UAV, IPPO style)
        self.uav_policies: List[UAVPolicyNetwork] = []
        self.uav_values: List[UAVValueNetwork] = []
        for _ in range(self.cfg.num_uavs):
            policy = UAVPolicyNetwork(
                obs_dim=self.encoder.uav_obs_dim,
                goal_dim=self.encoder.goal_dim,
                action_dim=3,
                hidden_dim=128
            ).to(self.device)
            value = UAVValueNetwork(
                obs_dim=self.encoder.uav_obs_dim,
                goal_dim=self.encoder.goal_dim,
                hidden_dim=128
            ).to(self.device)
            self.uav_policies.append(policy)
            self.uav_values.append(value)

        # Dynamically weighted queues (one per UAV)
        self.queues: List[DynamicallyWeightedQueue] = []

        # Temporal predictor
        self.temporal_predictor = TemporalPredictor(
            obs_dim=4, hidden_dim=64
        ).to(self.device)
        self.tp_trainer = TemporalPredictorTrainer(
            self.temporal_predictor, lr=1e-3
        )

        # Intrinsic reward computer
        self.intrinsic_rewarder = IntrinsicRewardComputer(
            omega=self.cfg.omega,
            world_size=self.cfg.world_size,
        )

        # PPO trainer
        self.ppo = PPOTrainer(lr=5e-4, batch_size=1200)

        # Optimizers
        self.opt_alloc_policy = torch.optim.Adam(self.alloc_policy.parameters(), lr=5e-4)
        self.opt_alloc_value = torch.optim.Adam(self.alloc_value.parameters(), lr=5e-4)
        self.opt_uav_policies = [
            torch.optim.Adam(p.parameters(), lr=5e-4)
            for p in self.uav_policies
        ]
        self.opt_uav_values = [
            torch.optim.Adam(v.parameters(), lr=5e-4)
            for v in self.uav_values
        ]

        # Buffers
        self.alloc_buffer = RolloutBuffer()
        self.uav_buffers = [RolloutBuffer() for _ in range(self.cfg.num_uavs)]

        # Trajectory tracking for temporal predictor
        self.emer_trajectories: Dict[int, List[np.ndarray]] = {}

        # Assignment tracking
        self.uav_goals: Dict[int, Optional[PoI]] = {}

    def reset(self):
        """Reset all buffers and queues for a new episode."""
        self.queues = [
            DynamicallyWeightedQueue(
                uav_id=i,
                max_length=self.cfg.queue_length,
                predictor=self.temporal_predictor
            )
            for i in range(self.cfg.num_uavs)
        ]
        self.alloc_buffer.clear()
        for buf in self.uav_buffers:
            buf.clear()
        self.emer_trajectories = {}
        self.uav_goals = {i: None for i in range(self.cfg.num_uavs)}

    def act(self, obs: dict, env: UAVCrowdsensingEnv) -> List[Tuple[float, float, float]]:
        """
        Algorithm 1 main loop: goal assignment + UAV execution.

        Args:
            obs: environment observation dict
            env: environment reference

        Returns:
            actions: list of (dx, dy, speed) for each UAV
        """
        uavs = env.uavs
        new_emer_pois = env.get_new_emergency_pois()

        # === High-level: Dynamic UAV Goal Assignment (Lines 5-7) ===
        for poi in new_emer_pois:
            # Build global state
            global_state = self.encoder.encode_global_state(obs['global'])
            state_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)

            # Select UAV for assignment (Line 6)
            with torch.no_grad():
                action, log_prob = self.alloc_policy.get_action(state_t)

            assigned_uav = action.item()

            # Insert into assigned UAV's queue (Line 7)
            uav_obs = obs.get(assigned_uav, {})
            uav_feat = self.encoder.encode_uav_obs(uav_obs) if uav_obs else np.zeros(self.encoder.uav_obs_dim)
            self.queues[assigned_uav].insert(poi, uav_feat)

            # Record for high-level buffer
            value = self.alloc_value(state_t).item()
            self.alloc_buffer.add(
                state=global_state,
                action=assigned_uav,
                reward=0.0,  # will be filled after episode
                value=value,
                log_prob=log_prob.item(),
                done=False,
            )

        # === Update queue priorities (Line 10) ===
        for i, queue in enumerate(self.queues):
            if i in obs:
                uav_feat = self.encoder.encode_uav_obs(obs[i])
                queue.update_priorities(
                    uav_feat,
                    lambda poi: self.encoder.poi_to_goal_features(poi)
                )
            queue.remove_handled()

        # === Low-level: Multi-task UAV Execution (Line 11) ===
        actions = []
        for i, uav in enumerate(uavs):
            if i not in obs:
                actions.append((0.0, 0.0, 0.5))
                continue

            uav_obs = obs[i]
            obs_feat = self.encoder.encode_uav_obs(uav_obs)

            # Get goal from queue (Line 11)
            goal_poi = self.queues[i].get_top_goal()
            self.uav_goals[i] = goal_poi

            if goal_poi is not None:
                goal_feat = self.encoder.poi_to_goal_features(goal_poi)
            else:
                # Default goal: cover nearby surveillance PoIs
                goal_feat = np.zeros(self.encoder.goal_dim, dtype=np.float32)

            # Store combined obs+goal for buffer
            combined = np.concatenate([obs_feat, goal_feat]).astype(np.float32)

            # Get action from policy network
            obs_t = torch.FloatTensor(obs_feat).unsqueeze(0).to(self.device)
            goal_t = torch.FloatTensor(goal_feat).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob = self.uav_policies[i].get_action(obs_t, goal_t)
                value = self.uav_values[i](obs_t, goal_t).item()

            dx, dy, speed = action[0].cpu().numpy()
            actions.append((float(dx), float(dy), float(speed)))

            # Record for low-level buffer
            self.uav_buffers[i].add(
                state=combined,
                action=action[0].cpu().numpy(),
                reward=0.0,  # will be filled
                value=value,
                log_prob=log_prob.item(),
                done=False,
            )

            # Track trajectory for temporal predictor
            if goal_poi is not None:
                pid = goal_poi.poi_id
                if pid not in self.emer_trajectories:
                    self.emer_trajectories[pid] = []
                self.emer_trajectories[pid].append(obs_feat[:4])

        return actions

    def record_rewards(self, rewards: List[float], done: bool, info: dict):
        """
        Record rewards after environment step.

        Args:
            rewards: low-level rewards from environment
            done: episode termination flag
            info: environment info dict
        """
        # Compute intrinsic rewards
        intrinsic_rewards = self.intrinsic_rewarder.compute_batch(
            [None] * self.cfg.num_uavs,  # placeholder
            self.uav_goals,
        )

        for i in range(self.cfg.num_uavs):
            if len(self.uav_buffers[i]) > 0:
                # Combine environmental + intrinsic reward (Eq. 13 + 15)
                env_r = rewards[i] if i < len(rewards) else 0.0
                intr_r = intrinsic_rewards[i] if i < len(intrinsic_rewards) else 0.0
                total_r = env_r + intr_r

                # Update last entry
                self.uav_buffers[i].rewards[-1] = total_r
                self.uav_buffers[i].dones[-1] = done

        # High-level reward (Eq. 12) - accumulated low-level reward
        if done and len(self.alloc_buffer) > 0:
            total_low_reward = sum(
                sum(buf.rewards) for buf in self.uav_buffers
            )
            avg_low = total_low_reward / max(self.cfg.num_uavs, 1)
            for j in range(len(self.alloc_buffer)):
                self.alloc_buffer.rewards[j] = avg_low
                self.alloc_buffer.dones[j] = done

    def train(self) -> Dict[str, float]:
        """
        Algorithm 1 Training Phase (Lines 20-24).

        Updates:
        1. High-level allocator parameters (Eq. 17-19)
        2. Temporal predictor parameters (Eq. 16)
        3. Low-level UAV parameters (Eq. 20)
        """
        metrics = {}

        # 1. Update high-level allocator (Line 22)
        alloc_metrics = self.ppo.update_high_level(
            self.alloc_policy, self.alloc_value,
            self.alloc_buffer,
            self.opt_alloc_policy, self.opt_alloc_value,
        )
        metrics.update({'alloc_' + k: v for k, v in alloc_metrics.items()})

        # 2. Update temporal predictor (Line 23)
        traj_list = list(self.emer_trajectories.values())
        tp_loss = self.tp_trainer.update(traj_list)
        metrics['tp_loss'] = tp_loss

        # 3. Update low-level UAV policies (Line 24)
        for i in range(self.cfg.num_uavs):
            uav_metrics = self.ppo.update_low_level(
                self.uav_policies[i], self.uav_values[i],
                self.uav_buffers[i],
                self.opt_uav_policies[i], self.opt_uav_values[i],
            )
            for k, v in uav_metrics.items():
                metrics[f'uav{i}_{k}'] = v

        return metrics
