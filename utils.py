"""
Utility functions for FlightRL project
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import json
import os
from datetime import datetime


class MetricsTracker:
    """Track training metrics over time"""
    
    def __init__(self):
        self.episode_returns = []
        self.episode_lengths = []
        self.success_rates = {0.0: deque(maxlen=100), 0.5: deque(maxlen=100), 1.0: deque(maxlen=100)}
        self.total_counts = {0.0: 0, 0.5: 0, 1.0: 0}
        self.success_counts = {0.0: 0, 0.5: 0, 1.0: 0}
        self.q_values = []
        self.losses = []
        
    def update(self, ep_return, ep_length, success, runway_type):
        """Update metrics with episode results"""
        self.episode_returns.append(ep_return)
        self.episode_lengths.append(ep_length)
        
        self.total_counts[runway_type] += 1
        if success:
            self.success_counts[runway_type] += 1
            self.success_rates[runway_type].append(1.0)
        else:
            self.success_rates[runway_type].append(0.0)
    
    def get_success_rates(self):
        """Get current success rates by runway type"""
        return {
            k: self.success_counts[k] / self.total_counts[k] 
            if self.total_counts[k] > 0 else 0.0
            for k in [0.0, 0.5, 1.0]
        }
    
    def get_recent_success_rates(self, window=100):
        """Get recent rolling success rates"""
        return {
            k: np.mean(list(self.success_rates[k])) if len(self.success_rates[k]) > 0 else 0.0
            for k in [0.0, 0.5, 1.0]
        }
    
    def get_average_return(self, window=100):
        """Get average return over recent episodes"""
        if len(self.episode_returns) == 0:
            return 0.0
        recent = self.episode_returns[-window:]
        return np.mean(recent)
    
    def save(self, filepath):
        """Save metrics to JSON file"""
        data = {
            'episode_returns': self.episode_returns,
            'episode_lengths': self.episode_lengths,
            'success_counts': {str(k): v for k, v in self.success_counts.items()},
            'total_counts': {str(k): v for k, v in self.total_counts.items()},
            'final_success_rates': {str(k): v for k, v in self.get_success_rates().items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath):
        """Load metrics from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.episode_returns = data['episode_returns']
        self.episode_lengths = data['episode_lengths']
        self.success_counts = {float(k): v for k, v in data['success_counts'].items()}
        self.total_counts = {float(k): v for k, v in data['total_counts'].items()}


def plot_training_curves(metrics_tracker, save_path=None):
    """Plot training curves for returns, lengths, and success rates"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Episode returns
    axes[0, 0].plot(metrics_tracker.episode_returns)
    axes[0, 0].set_title('Episode Returns')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].grid(True)
    
    # Moving average of returns
    if len(metrics_tracker.episode_returns) > 100:
        window = 100
        moving_avg = np.convolve(
            metrics_tracker.episode_returns, 
            np.ones(window)/window, 
            mode='valid'
        )
        axes[0, 1].plot(range(window-1, len(metrics_tracker.episode_returns)), moving_avg)
        axes[0, 1].set_title('Moving Average Return (window=100)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Return')
        axes[0, 1].grid(True)
    
    # Episode lengths
    axes[1, 0].plot(metrics_tracker.episode_lengths)
    axes[1, 0].set_title('Episode Lengths')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].grid(True)
    
    # Success rates by runway type
    success_rates = metrics_tracker.get_success_rates()
    axes[1, 1].bar(
        ['Dry', 'Wet', 'Icy'],
        [success_rates[0.0], success_rates[0.5], success_rates[1.0]]
    )
    axes[1, 1].set_title('Final Success Rates by Runway Type')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()


def compute_landing_statistics(env, agent, num_episodes=100):
    """Compute detailed landing statistics"""
    stats = {
        'successes': 0,
        'crashes': 0,
        'timeouts': 0,
        'altitude_at_landing': [],
        'speed_at_landing': [],
        'angle_at_landing': [],
        'distance_errors': [],
        'episode_lengths': []
    }
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.act(obs, epsilon=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        stats['episode_lengths'].append(env.steps)
        
        if info.get('success', False):
            stats['successes'] += 1
            stats['altitude_at_landing'].append(env.altitude)
            stats['speed_at_landing'].append(env.speed)
            stats['angle_at_landing'].append(env.angle)
            stats['distance_errors'].append(abs(env.distance))
        elif env.altitude <= 0 and env.distance > 0:
            stats['crashes'] += 1
        else:
            stats['timeouts'] += 1
    
    # Compute averages
    stats['success_rate'] = stats['successes'] / num_episodes
    stats['crash_rate'] = stats['crashes'] / num_episodes
    stats['timeout_rate'] = stats['timeouts'] / num_episodes
    
    if stats['altitude_at_landing']:
        stats['avg_altitude'] = np.mean(stats['altitude_at_landing'])
        stats['avg_speed'] = np.mean(stats['speed_at_landing'])
        stats['avg_angle'] = np.mean(stats['angle_at_landing'])
        stats['avg_distance_error'] = np.mean(stats['distance_errors'])
    
    stats['avg_episode_length'] = np.mean(stats['episode_lengths'])
    
    return stats


def save_experiment_config(args, save_dir):
    """Save experiment configuration to file"""
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, 'config.json')
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

