"""
Main training and evaluation script for DRL-MTUCS.

Usage:
    python main.py                    # Train DRL-MTUCS
    python main.py --mode eval        # Evaluate trained model
    python main.py --mode compare     # Compare with baselines
    python main.py --mode visualize   # Visualize trajectories
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List

from env.uav_env import UAVCrowdsensingEnv, SimConfig
from agents.drl_mtucs import DRLMTUCS
from agents.baselines import RandomBaseline, GreedyBaseline, mTSPBaseline


def train_drl_mtucs(config: SimConfig, num_episodes: int = 500,
                    log_interval: int = 10, save_dir: str = "checkpoints"):
    """
    Train DRL-MTUCS (Algorithm 1).

    Following the paper:
    - IPPO as base MADRL method
    - 3-layer MLP with 128 hidden states
    - Batch size 1200, Adam optimizer, lr=5e-4
    """
    os.makedirs(save_dir, exist_ok=True)

    env = UAVCrowdsensingEnv(config=config)
    agent = DRLMTUCS(config=config)

    print(f"Training DRL-MTUCS for {num_episodes} episodes")
    print(f"Config: {config.num_uavs} UAVs, {config.num_surv_pois} surv PoIs")
    print(f"Emer interval: {config.emer_interval}, ω: {config.omega}")
    print(f"Queue length: {config.queue_length}")
    print("=" * 60)

    all_metrics = []
    best_I = -np.inf

    for episode in range(num_episodes):
        obs = env.reset(seed=episode)
        agent.reset()

        episode_rewards = []
        done = False
        step = 0

        while not done:
            # Algorithm 1 Lines 5-11: act
            actions = agent.act(obs, env)

            # Algorithm 1 Line 14: environment interaction
            next_obs, rewards, done, info = env.step(actions)

            # Record rewards for training
            agent.record_rewards(rewards, done, info)

            episode_rewards.append(sum(rewards))
            obs = next_obs
            step += 1

        # Algorithm 1 Lines 20-24: train
        train_metrics = agent.train()

        # Log
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        metrics = {
            'episode': episode,
            'avg_reward': avg_reward,
            'total_reward': sum(episode_rewards),
            'steps': step,
            **info,
        }
        all_metrics.append(metrics)

        if (episode + 1) % log_interval == 0:
            recent = all_metrics[-log_interval:]
            avg_I = np.mean([m.get('I_index', 0) for m in recent])
            avg_I_surv = np.mean([m.get('I_surv', 0) for m in recent])
            avg_I_emer = np.mean([m.get('I_emer', 0) for m in recent])
            avg_r = np.mean([m['avg_reward'] for m in recent])

            print(f"Episode {episode+1:4d} | "
                  f"I={avg_I:.4f} | I_surv={avg_I_surv:.4f} | "
                  f"I_emer={avg_I_emer:.4f} | Reward={avg_r:.4f} | "
                  f"Steps={step}")

            # Save best model
            if avg_I > best_I:
                best_I = avg_I
                _save_checkpoint(agent, save_dir, "best")

    # Save final model and metrics
    _save_checkpoint(agent, save_dir, "final")
    _save_metrics(all_metrics, save_dir)
    _plot_training(all_metrics, save_dir)

    return agent, all_metrics


def evaluate(agent_or_name, config: SimConfig, num_episodes: int = 50,
             name: str = "DRL-MTUCS") -> Dict:
    """Evaluate an agent over multiple episodes."""
    if isinstance(agent_or_name, str):
        # Load from checkpoint
        agent = DRLMTUCS(config=config)
        _load_checkpoint(agent, agent_or_name)
    else:
        agent = agent_or_name

    env = UAVCrowdsensingEnv(config=config)

    all_info = []
    for ep in range(num_episodes):
        obs = env.reset(seed=1000 + ep)
        if hasattr(agent, 'reset'):
            agent.reset()

        done = False
        while not done:
            actions = agent.act(obs, env)
            obs, rewards, done, info = env.step(actions)

        all_info.append(info)

    # Aggregate
    results = {
        'name': name,
        'I_index_mean': np.mean([i['I_index'] for i in all_info]),
        'I_index_std': np.std([i['I_index'] for i in all_info]),
        'I_surv_mean': np.mean([i['I_surv'] for i in all_info]),
        'I_emer_mean': np.mean([i['I_emer'] for i in all_info]),
        'energy_ratio_mean': np.mean([i['energy_ratio'] for i in all_info]),
    }

    print(f"\n{'='*50}")
    print(f"Results for {name} ({num_episodes} episodes):")
    print(f"  Valid Task Handling Index (I): {results['I_index_mean']:.4f} ± {results['I_index_std']:.4f}")
    print(f"  Surveillance Handling Ratio:  {results['I_surv_mean']:.4f}")
    print(f"  Emergency Handling Ratio:     {results['I_emer_mean']:.4f}")
    print(f"  Energy Consumption Ratio:     {results['energy_ratio_mean']:.4f}")
    print(f"{'='*50}")

    return results


def compare_baselines(config: SimConfig, num_episodes: int = 50,
                      save_dir: str = "results"):
    """Compare DRL-MTUCS against all baselines (Section V-D)."""
    os.makedirs(save_dir, exist_ok=True)

    env = UAVCrowdsensingEnv(config=config)

    methods = {
        'Random': RandomBaseline(config.num_uavs),
        'Greedy': GreedyBaseline(config.num_uavs),
        'mTSP': mTSPBaseline(config.num_uavs),
    }

    # Try to load trained DRL-MTUCS
    ckpt_path = os.path.join("checkpoints", "best")
    if os.path.exists(ckpt_path):
        drl_agent = DRLMTUCS(config=config)
        _load_checkpoint(drl_agent, ckpt_path)
        methods['DRL-MTUCS'] = drl_agent

    all_results = {}
    for name, agent in methods.items():
        print(f"\nEvaluating {name}...")
        infos = []
        for ep in range(num_episodes):
            obs = env.reset(seed=2000 + ep)
            if hasattr(agent, 'reset'):
                agent.reset()
            done = False
            while not done:
                actions = agent.act(obs, env)
                obs, rewards, done, info = env.step(actions)
            infos.append(info)

        result = {
            'I_index': np.mean([i['I_index'] for i in infos]),
            'I_surv': np.mean([i['I_surv'] for i in infos]),
            'I_emer': np.mean([i['I_emer'] for i in infos]),
            'energy_ratio': np.mean([i['energy_ratio'] for i in infos]),
        }
        all_results[name] = result
        print(f"  I={result['I_index']:.4f} | I_surv={result['I_surv']:.4f} | "
              f"I_emer={result['I_emer']:.4f} | η={result['energy_ratio']:.4f}")

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'Method':<15} {'I_index':>10} {'I_surv':>10} {'I_emer':>10} {'η':>10}")
    print("-" * 70)
    for name, r in all_results.items():
        print(f"{name:<15} {r['I_index']:>10.4f} {r['I_surv']:>10.4f} "
              f"{r['I_emer']:>10.4f} {r['energy_ratio']:>10.4f}")
    print("=" * 70)

    # Save results
    with open(os.path.join(save_dir, "comparison.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Plot
    _plot_comparison(all_results, save_dir)

    return all_results


def visualize_trajectories(config: SimConfig, save_dir: str = "results"):
    """Visualize UAV trajectories (Section V-G)."""
    os.makedirs(save_dir, exist_ok=True)

    env = UAVCrowdsensingEnv(config=config)

    # Use greedy baseline for visualization
    agent = GreedyBaseline(config.num_uavs)

    obs = env.reset(seed=42)
    trajectories = {i: [(env.uavs[i].x, env.uavs[i].y)] for i in range(config.num_uavs)}
    emergency_snapshots = []
    done = False

    while not done:
        actions = agent.act(obs, env)
        obs, rewards, done, info = env.step(actions)

        for i in range(config.num_uavs):
            trajectories[i].append((env.uavs[i].x, env.uavs[i].y))

        # Snapshot at intervals
        if env.current_timeslot % 30 == 0:
            emergency_snapshots.append({
                'timeslot': env.current_timeslot,
                'uavs': [(u.x, u.y) for u in env.uavs],
                'emergencies': [(p.x, p.y, p.aoi, p.aoi_threshold) for p in env.emer_pois if p.active],
                'surv_handled': info.get('surv_handled', 0),
            })

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("UAV Trajectories - Multi-Task Crowdsensing", fontsize=14)

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    for idx, ax in enumerate(axes.flat):
        ax.set_xlim(0, config.world_size)
        ax.set_ylim(0, config.world_size)
        ax.set_aspect('equal')
        ax.set_title(f"Timeslot {(idx+1)*30}")

        # Plot surveillance PoIs
        surv_x = [p.x for p in env.surv_pois]
        surv_y = [p.y for p in env.surv_pois]
        ax.scatter(surv_x, surv_y, c='lightgray', s=5, alpha=0.5, label='Surveillance PoI')

        # Plot trajectories up to this time
        end_step = min((idx + 1) * 30, len(next(iter(trajectories.values()))))
        for i in range(config.num_uavs):
            traj = trajectories[i][:end_step]
            if len(traj) > 1:
                tx, ty = zip(*traj)
                ax.plot(tx, ty, c=colors[i % len(colors)], alpha=0.6, linewidth=1)
                ax.scatter([tx[-1]], [ty[-1]], c=colors[i % len(colors)],
                          s=80, marker='*', zorder=5)

        # Plot active emergencies
        if idx < len(emergency_snapshots):
            snap = emergency_snapshots[idx]
            for ex, ey, eaoi, eaoi_th in snap['emergencies']:
                ax.scatter([ex], [ey], c='red', s=100, marker='X', zorder=5)
                ax.annotate(f"AoI:{eaoi}/{eaoi_th}", (ex, ey), fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "trajectories.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Trajectory visualization saved to {save_dir}/trajectories.png")


def ablation_study(config: SimConfig, num_episodes: int = 30,
                   save_dir: str = "results"):
    """
    Ablation study (Section V-C):
    1. Without dynamically weighted queue (use distance instead)
    2. Without self-balancing intrinsic reward
    3. Full DRL-MTUCS
    """
    os.makedirs(save_dir, exist_ok=True)

    variants = {
        'ω=0.1': SimConfig(**{**config.__dict__, 'omega': 0.1}),
        'ω=0.3': SimConfig(**{**config.__dict__, 'omega': 0.3}),
        'ω=0.5': SimConfig(**{**config.__dict__, 'omega': 0.5}),
        'ω=0.7': SimConfig(**{**config.__dict__, 'omega': 0.7}),
        'ω=0.9': SimConfig(**{**config.__dict__, 'omega': 0.9}),
    }

    results = {}
    for name, cfg in variants.items():
        print(f"\nTesting {name}...")
        agent = DRLMTUCS(config=cfg)
        env = UAVCrowdsensingEnv(config=cfg)

        infos = []
        for ep in range(num_episodes):
            obs = env.reset(seed=3000 + ep)
            agent.reset()
            done = False
            while not done:
                actions = agent.act(obs, env)
                obs, rewards, done, info = env.step(actions)
                agent.record_rewards(rewards, done, info)
            agent.train()
            infos.append(info)

        r = {
            'I_index': np.mean([i['I_index'] for i in infos]),
            'I_surv': np.mean([i['I_surv'] for i in infos]),
            'I_emer': np.mean([i['I_emer'] for i in infos]),
        }
        results[name] = r
        print(f"  I={r['I_index']:.4f} | I_surv={r['I_surv']:.4f} | I_emer={r['I_emer']:.4f}")

    # Plot ω trade-off (like Fig. 4-5 in paper)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    omegas = [0.1, 0.3, 0.5, 0.7, 0.9]
    i_survs = [results[f'ω={w}']['I_surv'] for w in omegas]
    i_emers = [results[f'ω={w}']['I_emer'] for w in omegas]
    i_indexes = [results[f'ω={w}']['I_index'] for w in omegas]

    ax1.plot(omegas, i_survs, 'o-', label='I_surv', color='blue')
    ax1.plot(omegas, i_emers, 's-', label='I_emer', color='red')
    ax1.set_xlabel('ω (intrinsic reward weight)')
    ax1.set_ylabel('Valid Handling Ratio')
    ax1.legend()
    ax1.set_title('Task Trade-off vs ω')
    ax1.grid(True, alpha=0.3)

    ax2.plot(omegas, i_indexes, 'o-', label='I (index)', color='green')
    ax2.set_xlabel('ω (intrinsic reward weight)')
    ax2.set_ylabel('Valid Task Handling Index')
    ax2.legend()
    ax2.set_title('Overall Index vs ω')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ablation_omega.png"), dpi=150)
    plt.close()
    print(f"Ablation plot saved to {save_dir}/ablation_omega.png")

    return results


# === Utility Functions ===

def _save_checkpoint(agent: DRLMTUCS, save_dir: str, name: str):
    """Save model checkpoint."""
    path = os.path.join(save_dir, name)
    os.makedirs(path, exist_ok=True)
    torch.save({
        'alloc_policy': agent.alloc_policy.state_dict(),
        'alloc_value': agent.alloc_value.state_dict(),
        'uav_policies': [p.state_dict() for p in agent.uav_policies],
        'uav_values': [v.state_dict() for v in agent.uav_values],
        'temporal_predictor': agent.temporal_predictor.state_dict(),
    }, os.path.join(path, "model.pt"))


def _load_checkpoint(agent: DRLMTUCS, path: str):
    """Load model checkpoint."""
    import torch
    ckpt = torch.load(os.path.join(path, "model.pt"), map_location='cpu')
    agent.alloc_policy.load_state_dict(ckpt['alloc_policy'])
    agent.alloc_value.load_state_dict(ckpt['alloc_value'])
    for i, sd in enumerate(ckpt['uav_policies']):
        agent.uav_policies[i].load_state_dict(sd)
    for i, sd in enumerate(ckpt['uav_values']):
        agent.uav_values[i].load_state_dict(sd)
    agent.temporal_predictor.load_state_dict(ckpt['temporal_predictor'])


def _save_metrics(metrics: List[Dict], save_dir: str):
    """Save training metrics to JSON."""
    with open(os.path.join(save_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)


def _plot_training(metrics: List[Dict], save_dir: str):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    episodes = [m['episode'] for m in metrics]

    # Smooth with window
    def smooth(arr, w=10):
        return np.convolve(arr, np.ones(w)/w, mode='valid')

    ax = axes[0, 0]
    rewards = [m['avg_reward'] for m in metrics]
    ax.plot(smooth(rewards), alpha=0.7)
    ax.set_title('Average Reward')
    ax.set_xlabel('Episode')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    I_index = [m.get('I_index', 0) for m in metrics]
    ax.plot(smooth(I_index), alpha=0.7, color='green')
    ax.set_title('Valid Task Handling Index (I)')
    ax.set_xlabel('Episode')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    I_surv = [m.get('I_surv', 0) for m in metrics]
    I_emer = [m.get('I_emer', 0) for m in metrics]
    ax.plot(smooth(I_surv), alpha=0.7, label='I_surv', color='blue')
    ax.plot(smooth(I_emer), alpha=0.7, label='I_emer', color='red')
    ax.set_title('Valid Handling Ratios')
    ax.set_xlabel('Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    eta = [m.get('energy_ratio', 0) for m in metrics]
    ax.plot(smooth(eta), alpha=0.7, color='orange')
    ax.set_title('Energy Consumption Ratio (η)')
    ax.set_xlabel('Episode')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()


def _plot_comparison(results: Dict, save_dir: str):
    """Plot comparison bar chart."""
    names = list(results.keys())
    I_surv = [results[n]['I_surv'] for n in names]
    I_emer = [results[n]['I_emer'] for n in names]
    I_index = [results[n]['I_index'] for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, I_surv, width, label='I_surv', color='#377eb8')
    ax.bar(x, I_emer, width, label='I_emer', color='#e41a1c')
    ax.bar(x + width, I_index, width, label='I_index', color='#4daf4a')

    ax.set_ylabel('Score')
    ax.set_title('Baseline Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison.png"), dpi=150)
    plt.close()


# === Main Entry Point ===

import torch

def main():
    parser = argparse.ArgumentParser(description='DRL-MTUCS: Multi-Task UAV Crowdsensing')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'compare', 'visualize', 'ablation'])
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--num_uavs', type=int, default=4)
    parser.add_argument('--num_surv_pois', type=int, default=300)
    parser.add_argument('--emer_interval', type=int, default=6)
    parser.add_argument('--surv_aoi', type=int, default=35)
    parser.add_argument('--emer_aoi', type=int, default=20)
    parser.add_argument('--omega', type=float, default=0.7)
    parser.add_argument('--queue_length', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results')
    args = parser.parse_args()

    config = SimConfig(
        num_uavs=args.num_uavs,
        num_surv_pois=args.num_surv_pois,
        emer_interval=args.emer_interval,
        surv_aoi_threshold=args.surv_aoi,
        emer_aoi_threshold=args.emer_aoi,
        omega=args.omega,
        queue_length=args.queue_length,
    )

    if args.mode == 'train':
        train_drl_mtucs(config, num_episodes=args.episodes, save_dir=args.save_dir)

    elif args.mode == 'eval':
        ckpt = os.path.join(args.save_dir, "best")
        if not os.path.exists(ckpt):
            print("No checkpoint found. Train first with --mode train")
            return
        evaluate(ckpt, config, num_episodes=50)

    elif args.mode == 'compare':
        compare_baselines(config, num_episodes=50, save_dir=args.save_dir)

    elif args.mode == 'visualize':
        visualize_trajectories(config, save_dir=args.save_dir)

    elif args.mode == 'ablation':
        ablation_study(config, num_episodes=30, save_dir=args.save_dir)


if __name__ == '__main__':
    main()
