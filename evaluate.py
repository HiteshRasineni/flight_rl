import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from envs.flight_env import FlightEnv
from agents.dqn_agent import DQNAgent

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--save_plots', type=str, default='./evaluation_plots')
    p.add_argument('--show_plots', action='store_true')
    return p.parse_args()

def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'

    env = FlightEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, device=device)
    agent.load(args.checkpoint)

    # Comprehensive metrics tracking
    success_counts = {0.0: 0, 0.5: 0, 1.0: 0}
    total_counts = {0.0: 0, 0.5: 0, 1.0: 0}
    
    # Detailed statistics
    episode_rewards = []
    episode_lengths = []
    landing_altitudes = []
    landing_speeds = []
    landing_angles = []
    landing_distances = []
    
    # Track by runway type
    metrics_by_runway = {
        0.0: {'rewards': [], 'lengths': [], 'altitudes': [], 'speeds': [], 'angles': [], 'distances': []},
        0.5: {'rewards': [], 'lengths': [], 'altitudes': [], 'speeds': [], 'angles': [], 'distances': []},
        1.0: {'rewards': [], 'lengths': [], 'altitudes': [], 'speeds': [], 'angles': [], 'distances': []}
    }
    
    crashes = 0
    timeouts = 0
    successes = 0

    print(f"Evaluating agent with {args.episodes} episodes...")
    print("-" * 60)

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        runway_type = env.runway_condition
        ep_steps = 0
        ep_reward = 0.0

        while not done:
            action = agent.act(obs, epsilon=0.0)  # fully greedy
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = next_obs
            ep_steps += 1
            ep_reward += reward

        # Record metrics
        total_counts[runway_type] += 1
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)
        
        success = info.get("success", False)
        if success:
            success_counts[runway_type] += 1
            successes += 1
            landing_altitudes.append(env.altitude)
            landing_speeds.append(env.speed)
            landing_angles.append(abs(env.angle))
            landing_distances.append(abs(env.distance))
            
            # Record by runway type
            metrics_by_runway[runway_type]['altitudes'].append(env.altitude)
            metrics_by_runway[runway_type]['speeds'].append(env.speed)
            metrics_by_runway[runway_type]['angles'].append(abs(env.angle))
            metrics_by_runway[runway_type]['distances'].append(abs(env.distance))
        elif env.altitude <= 0 and env.distance > 0:
            crashes += 1
        else:
            timeouts += 1
        
        # Record all episodes by runway type
        metrics_by_runway[runway_type]['rewards'].append(ep_reward)
        metrics_by_runway[runway_type]['lengths'].append(ep_steps)

        if ep % 10 == 0 or ep == args.episodes:
            print(f"[Ep {ep}/{args.episodes}] Steps: {ep_steps:3d} | Reward: {ep_reward:7.2f} | "
                  f"Runway: {runway_type} | Success: {success}")

    # Calculate statistics
    success_rates = {k: (success_counts[k]/total_counts[k] if total_counts[k]>0 else 0.0) 
                     for k in success_counts}
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Statistics:")
    print(f"  Total Episodes: {args.episodes}")
    print(f"  Successful Landings: {successes} ({100*successes/args.episodes:.1f}%)")
    print(f"  Crashes: {crashes} ({100*crashes/args.episodes:.1f}%)")
    print(f"  Timeouts: {timeouts} ({100*timeouts/args.episodes:.1f}%)")
    print(f"  Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    print(f"\nSuccess Rates by Runway Condition:")
    runway_names = {0.0: "Dry", 0.5: "Wet", 1.0: "Icy"}
    for k, rate in success_rates.items():
        count = success_counts[k]
        total = total_counts[k]
        print(f"  {runway_names[k]}: {count}/{total} = {100*rate:.1f}%")
    
    if landing_altitudes:
        print(f"\nLanding Statistics (Successful Landings Only):")
        print(f"  Altitude: {np.mean(landing_altitudes):.1f} ± {np.std(landing_altitudes):.1f} m")
        print(f"  Speed: {np.mean(landing_speeds):.1f} ± {np.std(landing_speeds):.1f} knots")
        print(f"  Angle: {np.mean(landing_angles):.1f} ± {np.std(landing_angles):.1f} degrees")
        print(f"  Distance Error: {np.mean(landing_distances):.1f} ± {np.std(landing_distances):.1f} m")

    # Create plots
    os.makedirs(args.save_plots, exist_ok=True)
    create_evaluation_plots(
        success_rates, success_counts, total_counts,
        episode_rewards, episode_lengths,
        landing_altitudes, landing_speeds, landing_angles, landing_distances,
        metrics_by_runway, crashes, timeouts, successes,
        args.episodes, args.save_plots, args.show_plots
    )
    
    print(f"\nPlots saved to: {args.save_plots}")

def create_evaluation_plots(success_rates, success_counts, total_counts,
                            episode_rewards, episode_lengths,
                            landing_altitudes, landing_speeds, landing_angles, landing_distances,
                            metrics_by_runway, crashes, timeouts, successes,
                            num_episodes, save_dir, show_plots):
    """Create comprehensive evaluation plots"""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Success Rates by Runway Condition
    fig, ax = plt.subplots(figsize=(8, 6))
    runway_names = ['Dry', 'Wet', 'Icy']
    rates = [success_rates[0.0], success_rates[0.5], success_rates[1.0]]
    colors = ['#2ecc71', '#3498db', '#95a5a6']
    
    bars = ax.bar(runway_names, rates, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
    ax.set_xlabel('Runway Condition', fontsize=12, fontweight='bold')
    ax.set_title('Landing Success Rate by Runway Condition', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        height = bar.get_height()
        count = success_counts[[0.0, 0.5, 1.0][i]]
        total = total_counts[[0.0, 0.5, 1.0][i]]
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{100*rate:.1f}%\n({count}/{total})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '1_success_rates.png'), dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()
    
    # 2. Overall Performance Distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Episode Rewards Distribution
    axes[0, 0].hist(episode_rewards, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(episode_rewards), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(episode_rewards):.2f}')
    axes[0, 0].set_xlabel('Episode Reward', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Episode Rewards Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode Lengths Distribution
    axes[0, 1].hist(episode_lengths, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(episode_lengths), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(episode_lengths):.1f}')
    axes[0, 1].set_xlabel('Episode Length (steps)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Episode Lengths Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Episode Rewards Over Time
    axes[1, 0].plot(episode_rewards, alpha=0.6, color='#3498db', linewidth=1)
    if len(episode_rewards) >= 10:
        window = min(20, len(episode_rewards) // 5)
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(range(window-1, len(episode_rewards)), moving_avg, 
                       color='red', linewidth=2, label=f'Moving Avg (window={window})')
        axes[1, 0].legend()
    axes[1, 0].set_xlabel('Episode', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Reward', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Episode Rewards Over Time', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Outcome Pie Chart
    outcome_counts = [successes, crashes, timeouts]
    outcome_labels = ['Success', 'Crash', 'Timeout']
    colors_pie = ['#2ecc71', '#e74c3c', '#f39c12']
    axes[1, 1].pie(outcome_counts, labels=outcome_labels, colors=colors_pie, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[1, 1].set_title('Landing Outcomes', fontsize=12, fontweight='bold')
    
    plt.suptitle('Overall Performance Metrics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2_overall_performance.png'), dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()
    
    # 3. Landing Statistics (for successful landings only)
    if landing_altitudes:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Landing Altitude Distribution
        axes[0, 0].hist(landing_altitudes, bins=20, color='#16a085', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(25, color='green', linestyle='--', linewidth=2, label='Optimal (25m)')
        axes[0, 0].axvline(np.mean(landing_altitudes), color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {np.mean(landing_altitudes):.1f}m')
        axes[0, 0].set_xlabel('Altitude (m)', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Landing Altitude Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Landing Speed Distribution
        axes[0, 1].hist(landing_speeds, bins=20, color='#2980b9', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(150, color='green', linestyle='--', linewidth=2, label='Optimal (150 kn)')
        axes[0, 1].axvline(np.mean(landing_speeds), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(landing_speeds):.1f} kn')
        axes[0, 1].set_xlabel('Speed (knots)', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Landing Speed Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Landing Angle Distribution
        axes[1, 0].hist(landing_angles, bins=20, color='#8e44ad', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(0, color='green', linestyle='--', linewidth=2, label='Optimal (0°)')
        axes[1, 0].axvline(np.mean(landing_angles), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(landing_angles):.1f}°')
        axes[1, 0].set_xlabel('Pitch Angle (degrees)', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Landing Angle Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Landing Distance Error
        axes[1, 1].hist(landing_distances, bins=20, color='#e67e22', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(0, color='green', linestyle='--', linewidth=2, label='Perfect (0m)')
        axes[1, 1].axvline(np.mean(landing_distances), color='red', linestyle='--', linewidth=2,
                         label=f'Mean: {np.mean(landing_distances):.1f}m')
        axes[1, 1].set_xlabel('Distance Error (m)', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Landing Distance Error Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Landing Statistics (Successful Landings Only)', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '3_landing_statistics.png'), dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close()
    
    # 4. Performance Comparison by Runway Type
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Average Rewards by Runway
    rewards_by_runway = [np.mean(metrics_by_runway[k]['rewards']) if metrics_by_runway[k]['rewards'] else 0
                        for k in [0.0, 0.5, 1.0]]
    std_by_runway = [np.std(metrics_by_runway[k]['rewards']) if metrics_by_runway[k]['rewards'] else 0
                    for k in [0.0, 0.5, 1.0]]
    
    bars = axes[0, 0].bar(runway_names, rewards_by_runway, yerr=std_by_runway, 
                         color=colors, alpha=0.8, edgecolor='black', linewidth=2, capsize=5)
    axes[0, 0].set_ylabel('Average Reward', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Runway Condition', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Average Reward by Runway Condition', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rewards_by_runway):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + std_by_runway[rewards_by_runway.index(val)] + 2,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Average Episode Length by Runway
    lengths_by_runway = [np.mean(metrics_by_runway[k]['lengths']) if metrics_by_runway[k]['lengths'] else 0
                         for k in [0.0, 0.5, 1.0]]
    std_lengths = [np.std(metrics_by_runway[k]['lengths']) if metrics_by_runway[k]['lengths'] else 0
                  for k in [0.0, 0.5, 1.0]]
    
    bars = axes[0, 1].bar(runway_names, lengths_by_runway, yerr=std_lengths,
                         color=colors, alpha=0.8, edgecolor='black', linewidth=2, capsize=5)
    axes[0, 1].set_ylabel('Average Episode Length', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Runway Condition', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Average Episode Length by Runway Condition', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, lengths_by_runway):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + std_lengths[lengths_by_runway.index(val)] + 2,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Landing Altitude by Runway (if available)
    if any(metrics_by_runway[k]['altitudes'] for k in [0.0, 0.5, 1.0]):
        altitudes_by_runway = [np.mean(metrics_by_runway[k]['altitudes']) if metrics_by_runway[k]['altitudes'] else 0
                              for k in [0.0, 0.5, 1.0]]
        std_altitudes = [np.std(metrics_by_runway[k]['altitudes']) if metrics_by_runway[k]['altitudes'] else 0
                        for k in [0.0, 0.5, 1.0]]
        
        bars = axes[1, 0].bar(runway_names, altitudes_by_runway, yerr=std_altitudes,
                             color=colors, alpha=0.8, edgecolor='black', linewidth=2, capsize=5)
        axes[1, 0].axhline(25, color='green', linestyle='--', linewidth=2, label='Optimal (25m)')
        axes[1, 0].set_ylabel('Average Landing Altitude (m)', fontsize=11, fontweight='bold')
        axes[1, 0].set_xlabel('Runway Condition', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Average Landing Altitude by Runway Condition', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Landing Speed by Runway (if available)
    if any(metrics_by_runway[k]['speeds'] for k in [0.0, 0.5, 1.0]):
        speeds_by_runway = [np.mean(metrics_by_runway[k]['speeds']) if metrics_by_runway[k]['speeds'] else 0
                           for k in [0.0, 0.5, 1.0]]
        std_speeds = [np.std(metrics_by_runway[k]['speeds']) if metrics_by_runway[k]['speeds'] else 0
                     for k in [0.0, 0.5, 1.0]]
        
        bars = axes[1, 1].bar(runway_names, speeds_by_runway, yerr=std_speeds,
                             color=colors, alpha=0.8, edgecolor='black', linewidth=2, capsize=5)
        axes[1, 1].axhline(150, color='green', linestyle='--', linewidth=2, label='Optimal (150 kn)')
        axes[1, 1].set_ylabel('Average Landing Speed (knots)', fontsize=11, fontweight='bold')
        axes[1, 1].set_xlabel('Runway Condition', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Average Landing Speed by Runway Condition', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Performance Comparison by Runway Condition', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '4_runway_comparison.png'), dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()
    
    print("Generated evaluation plots:")
    print("  1. Success rates by runway condition")
    print("  2. Overall performance metrics")
    if landing_altitudes:
        print("  3. Landing statistics (successful landings)")
    print("  4. Performance comparison by runway condition")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
