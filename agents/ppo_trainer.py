"""
PPO Trainer for DRL-MTUCS (Section IV-C)

Implements the training pipeline with:
- High-level allocator PPO update (Eq. 17-19)
- Low-level UAV policy PPO update (Eq. 20)
- GAE (Generalized Advantage Estimation)

DRL-MTUCS works with any actor-critic MADRL method;
we use IPPO (Independent PPO) as the base, following the paper.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque

from agents.networks import (
    UAVPolicyNetwork, UAVValueNetwork,
    HighLevelAllocatorNetwork, HighLevelValueNetwork,
    FeatureEncoder
)


class RolloutBuffer:
    """Stores rollout data for PPO updates."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.next_states = []

    def add(self, state, action, reward, value, log_prob, done, next_state=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.next_states.append(next_state)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


class PPOTrainer:
    """
    PPO trainer implementing Eq. 17-20 from the paper.

    Supports both high-level (discrete actions) and
    low-level (continuous actions) policy updates.
    """

    def __init__(
        self,
        clip_eps: float = 0.2,
        value_clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        critic_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        lr: float = 5e-4,
        batch_size: int = 1200,  # paper Table III
    ):
        self.clip_eps = clip_eps
        self.value_clip_eps = value_clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.batch_size = batch_size

    def compute_gae(self, rewards: List[float], values: List[float],
                    dones: List[bool], next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Returns:
            advantages: GAE advantages
            returns: discounted returns
        """
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update_low_level(
        self,
        policy: UAVPolicyNetwork,
        value_net: UAVValueNetwork,
        buffer: RolloutBuffer,
        optimizer_policy: torch.optim.Optimizer,
        optimizer_value: torch.optim.Optimizer,
        epochs: int = 4,
    ) -> Dict[str, float]:
        """
        Update low-level UAV policy (Eq. 20).

        L_low = Σ_u [L(θ_u) + c_critic L(φ_u) + c_entropy H(π_u)]
        """
        if len(buffer) == 0:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}

        # Prepare data
        obs_batch = torch.FloatTensor(np.array(buffer.states))
        action_batch = torch.FloatTensor(np.array(buffer.actions))
        old_log_prob_batch = torch.FloatTensor(np.array(buffer.log_probs))
        rewards = np.array(buffer.rewards)
        values = np.array(buffer.values)
        dones = np.array(buffer.dones)

        # Compute GAE
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns)

        # Normalize advantages
        if advantages_tensor.std() > 1e-8:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        total_p_loss = 0
        total_v_loss = 0
        total_entropy = 0

        for _ in range(epochs):
            # Split obs into obs and goal (obs_batch is concatenated)
            obs_part = obs_batch[:, :obs_batch.shape[1] - 8]
            goal_part = obs_batch[:, -8:]

            # Policy update (Eq. 17)
            new_log_prob, entropy = policy.evaluate_action(obs_part, goal_part, action_batch)
            ratio = (new_log_prob - old_log_prob_batch).exp()

            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()

            # Entropy bonus
            entropy_loss = -entropy.mean()

            total_loss = policy_loss + self.entropy_coef * entropy_loss

            optimizer_policy.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
            optimizer_policy.step()

            # Value update (Eq. 18)
            new_values = value_net(obs_part, goal_part).squeeze()
            value_pred_clipped = torch.FloatTensor(values) + torch.clamp(
                new_values - torch.FloatTensor(values),
                -self.value_clip_eps, self.value_clip_eps
            )
            v_loss1 = F.mse_loss(new_values, returns_tensor)
            v_loss2 = F.mse_loss(value_pred_clipped, returns_tensor)
            value_loss = torch.max(v_loss1, v_loss2)

            optimizer_value.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), self.max_grad_norm)
            optimizer_value.step()

            total_p_loss += policy_loss.item()
            total_v_loss += value_loss.item()
            total_entropy += entropy.mean().item()

        return {
            'policy_loss': total_p_loss / epochs,
            'value_loss': total_v_loss / epochs,
            'entropy': total_entropy / epochs,
        }

    def update_high_level(
        self,
        policy: HighLevelAllocatorNetwork,
        value_net: HighLevelValueNetwork,
        buffer: RolloutBuffer,
        optimizer_policy: torch.optim.Optimizer,
        optimizer_value: torch.optim.Optimizer,
        epochs: int = 4,
    ) -> Dict[str, float]:
        """
        Update high-level allocator policy (Eq. 17-19).

        Same PPO update but with discrete actions.
        """
        if len(buffer) == 0:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}

        states = torch.FloatTensor(np.array(buffer.states))
        actions = torch.LongTensor(np.array(buffer.actions))
        old_log_probs = torch.FloatTensor(np.array(buffer.log_probs))
        rewards = np.array(buffer.rewards)
        values = np.array(buffer.values)
        dones = np.array(buffer.dones)

        advantages, returns = self.compute_gae(rewards, values, dones)
        adv_t = torch.FloatTensor(advantages)
        ret_t = torch.FloatTensor(returns)

        if adv_t.std() > 1e-8:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_p_loss = 0
        total_v_loss = 0
        total_entropy = 0

        for _ in range(epochs):
            new_log_prob, entropy = policy.evaluate_action(states, actions)
            ratio = (new_log_prob - old_log_probs).exp()

            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
            p_loss = -torch.min(surr1, surr2).mean()
            e_loss = -entropy.mean()

            loss = p_loss + self.entropy_coef * e_loss
            optimizer_policy.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
            optimizer_policy.step()

            # Value
            new_v = value_net(states).squeeze()
            pred_clip = torch.FloatTensor(values) + torch.clamp(
                new_v - torch.FloatTensor(values),
                -self.value_clip_eps, self.value_clip_eps
            )
            vl1 = F.mse_loss(new_v, ret_t)
            vl2 = F.mse_loss(pred_clip, ret_t)
            v_loss = torch.max(vl1, vl2)

            optimizer_value.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), self.max_grad_norm)
            optimizer_value.step()

            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()
            total_entropy += entropy.mean().item()

        return {
            'policy_loss': total_p_loss / epochs,
            'value_loss': total_v_loss / epochs,
            'entropy': total_entropy / epochs,
        }
