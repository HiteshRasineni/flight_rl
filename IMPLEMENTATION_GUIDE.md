# Implementation Guide for Enhanced Features

## Quick Start: Using Enhanced Components

### 1. Enhanced Environment with Wind Effects

```python
from envs.flight_env_enhanced import EnhancedFlightEnv

# Create environment with wind
env = EnhancedFlightEnv(enable_wind=True, wind_severity=1.0)

# Train with enhanced environment
obs, _ = env.reset()
# ... training code ...
```

### 2. Enhanced DQN Agent (Double + Dueling)

```python
from agents.dqn_agent_enhanced import EnhancedDQNAgent

# Create enhanced agent
agent = EnhancedDQNAgent(
    obs_dim=8,  # Enhanced env has 8 dims
    n_actions=5,
    lr=5e-4,
    use_double=True,  # Enable Double DQN
    use_dueling=True  # Enable Dueling architecture
)
```

### 3. Using Utility Functions

```python
from utils import MetricsTracker, plot_training_curves, compute_landing_statistics

# Track metrics during training
tracker = MetricsTracker()

for episode in range(num_episodes):
    # ... training ...
    tracker.update(ep_return, ep_length, success, runway_type)

# Plot training curves
plot_training_curves(tracker, save_path='training_curves.png')

# Compute detailed statistics
stats = compute_landing_statistics(env, agent, num_episodes=100)
print(f"Success rate: {stats['success_rate']}")
```

### 4. Curriculum Learning

```python
from curriculum_learning import CurriculumFlightEnv

# Create curriculum environment
env = CurriculumFlightEnv(use_enhanced=False)

for episode in range(num_episodes):
    obs, _ = env.reset()
    # ... training ...
    env.update_curriculum(steps=episode_length)
```

## Example: Enhanced Training Script

Here's how to modify your training script to use enhanced features:

```python
import torch
from envs.flight_env_enhanced import EnhancedFlightEnv
from agents.dqn_agent_enhanced import EnhancedDQNAgent
from agents.replay_buffer import ReplayBuffer
from utils import MetricsTracker, save_experiment_config, set_seed
import argparse

def train_enhanced(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Use enhanced environment
    env = EnhancedFlightEnv(enable_wind=args.enable_wind, wind_severity=args.wind_severity)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    # Use enhanced agent
    agent = EnhancedDQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        lr=args.lr,
        use_double=True,
        use_dueling=True,
        device=device
    )
    
    buffer = ReplayBuffer(args.buffer_size, (obs_dim,))
    tracker = MetricsTracker()
    
    # Save experiment config
    save_experiment_config(args, args.save_dir)
    
    # Training loop
    epsilon = args.eps_start
    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        ep_length = 0
        done = False
        
        while not done:
            action = agent.act(obs, epsilon=epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_return += reward
            ep_length += 1
            
            if len(buffer) >= args.batch_size:
                batch = buffer.sample(args.batch_size)
                loss = agent.update(batch, args.batch_size)
                tracker.losses.append(loss)
            
            # Update epsilon
            epsilon = max(args.eps_end, epsilon - (args.eps_start - args.eps_end) / args.eps_decay_steps)
        
        # Update metrics
        tracker.update(ep_return, ep_length, info.get('success', False), env.runway_condition)
        
        # Periodic saves
        if ep % args.save_every == 0:
            agent.save(f"{args.save_dir}/dqn_ep{ep}.pt")
            tracker.save(f"{args.save_dir}/metrics_ep{ep}.json")
    
    # Final saves
    agent.save(f"{args.save_dir}/dqn_final.pt")
    tracker.save(f"{args.save_dir}/metrics_final.json")
    plot_training_curves(tracker, save_path=f"{args.save_dir}/training_curves.png")
    
    print("Training complete!")
    print(f"Final success rates: {tracker.get_success_rates()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--buffer_size', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.05)
    parser.add_argument('--eps_decay_steps', type=int, default=100000)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--enable_wind', action='store_true')
    parser.add_argument('--wind_severity', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    train_enhanced(args)
```

## Comparison Scripts

### Compare Standard vs Enhanced DQN

```python
from envs.flight_env import FlightEnv
from agents.dqn_agent import DQNAgent
from agents.dqn_agent_enhanced import EnhancedDQNAgent
from utils import compute_landing_statistics

# Test standard DQN
env_std = FlightEnv()
agent_std = DQNAgent(obs_dim=5, n_actions=5)
agent_std.load('checkpoints/dqn_final.pt')

stats_std = compute_landing_statistics(env_std, agent_std, num_episodes=100)

# Test enhanced DQN
env_enh = FlightEnv()
agent_enh = EnhancedDQNAgent(obs_dim=5, n_actions=5, use_double=True, use_dueling=True)
agent_enh.load('checkpoints/dqn_enhanced_final.pt')

stats_enh = compute_landing_statistics(env_enh, agent_enh, num_episodes=100)

print(f"Standard DQN Success Rate: {stats_std['success_rate']:.2%}")
print(f"Enhanced DQN Success Rate: {stats_enh['success_rate']:.2%}")
```

### Compare with/without Wind

```python
from envs.flight_env_enhanced import EnhancedFlightEnv
from utils import compute_landing_statistics

# Without wind
env_no_wind = EnhancedFlightEnv(enable_wind=False)
stats_no_wind = compute_landing_statistics(env_no_wind, agent, num_episodes=100)

# With wind
env_wind = EnhancedFlightEnv(enable_wind=True, wind_severity=1.0)
stats_wind = compute_landing_statistics(env_wind, agent, num_episodes=100)

print(f"Without Wind: {stats_no_wind['success_rate']:.2%}")
print(f"With Wind: {stats_wind['success_rate']:.2%}")
```

## Experiments for Paper

### Experiment 1: Standard Training
```bash
python train_dqn.py --episodes 10000 --save_dir ./results/standard
```

### Experiment 2: Enhanced DQN
```python
# Use enhanced agent with Double DQN + Dueling
# Modify train_dqn.py to use EnhancedDQNAgent
```

### Experiment 3: Enhanced Environment
```python
# Use EnhancedFlightEnv instead of FlightEnv
```

### Experiment 4: Curriculum Learning
```python
# Use CurriculumFlightEnv
```

### Experiment 5: Wind Effects
```python
# Train on EnhancedFlightEnv with wind enabled
```

## Evaluation Scripts

### Comprehensive Evaluation

```python
from utils import compute_landing_statistics
import numpy as np

# Run multiple seeds for statistical significance
seeds = [42, 123, 456, 789, 1011]
success_rates = []

for seed in seeds:
    set_seed(seed)
    stats = compute_landing_statistics(env, agent, num_episodes=100)
    success_rates.append(stats['success_rate'])

mean_sr = np.mean(success_rates)
std_sr = np.std(success_rates)

print(f"Success Rate: {mean_sr:.2%} ± {std_sr:.2%}")
```

### Ablation Studies

```python
# Test different components
configs = [
    {'use_double': False, 'use_dueling': False},  # Standard DQN
    {'use_double': True, 'use_dueling': False},   # Double DQN
    {'use_double': False, 'use_dueling': True},   # Dueling DQN
    {'use_double': True, 'use_dueling': True},    # Both
]

for config in configs:
    agent = EnhancedDQNAgent(obs_dim=5, n_actions=5, **config)
    # Train and evaluate
```

## Tips

1. **Start with Standard Components**: First ensure standard DQN works well
2. **Gradually Add Features**: Add one enhancement at a time to measure impact
3. **Use Metrics Tracker**: Always use `MetricsTracker` for comprehensive logging
4. **Save Configs**: Use `save_experiment_config` to track hyperparameters
5. **Multiple Seeds**: Run multiple seeds for statistical significance
6. **Compare Baselines**: Always compare with simpler baselines

## Troubleshooting

### Issue: Enhanced environment has different observation space
**Solution**: Make sure agent's `obs_dim` matches environment's observation space shape.

### Issue: Training instability with enhanced features
**Solution**: Reduce learning rate, increase buffer size, or disable some enhancements initially.

### Issue: Curriculum learning not improving
**Solution**: Adjust curriculum schedule parameters or use different schedule type.

