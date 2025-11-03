# FlightRL: Deep Reinforcement Learning for Aircraft Landing

A deep reinforcement learning framework for autonomous aircraft landing using Deep Q-Networks (DQN). This project implements a simplified flight environment with runway condition variability and trains an RL agent to perform safe landings under different conditions.

## Overview

This project uses Deep Q-Networks (DQN) to learn optimal landing strategies for aircraft. The agent learns to control throttle and pitch to safely land on runways with varying conditions (dry, wet, icy).

## Features

- **DQN-based Agent**: Deep Q-Network with target network and experience replay
- **Flight Environment**: Simplified physics-based flight model with:
  - Altitude, speed, distance, and pitch angle tracking
  - Variable runway conditions (dry, wet, icy)
  - Realistic landing constraints
- **Visualization Tools**: Pygame-based real-time visualization and matplotlib trajectory plots
- **Evaluation Metrics**: Success rate tracking by runway condition type

## Project Structure

```
flight_rl/
├── agents/
│   ├── dqn_agent.py       # DQN agent implementation
│   └── replay_buffer.py   # Experience replay buffer
├── envs/
│   └── flight_env.py      # Flight landing environment
├── checkpoints/           # Saved model checkpoints
├── train_dqn.py          # Training script
├── evaluate.py            # Model evaluation script
├── visualize.py           # Matplotlib trajectory visualization
├── pygame_visualize.py    # Real-time pygame visualization
└── utils.py              # Utility functions
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a DQN agent:
```bash
python train_dqn.py --episodes 10000 --lr 5e-4 --buffer_size 50000
```

Options:
- `--episodes`: Number of training episodes (default: 10000)
- `--lr`: Learning rate (default: 5e-4)
- `--buffer_size`: Replay buffer size (default: 50000)
- `--batch_size`: Batch size for training (default: 64)
- `--eps_start`: Initial exploration rate (default: 1.0)
- `--eps_end`: Final exploration rate (default: 0.05)
- `--target_update`: Steps between target network updates (default: 500)
- `--save_every`: Episodes between checkpoint saves (default: 500)
- `--cpu`: Force CPU usage

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py --checkpoint ./checkpoints/dqn_final.pt --episodes 100
```

### Visualization

Visualize trajectories with matplotlib:
```bash
python visualize.py --checkpoint ./checkpoints/dqn_final.pt --episodes 5
```

Real-time visualization with pygame:
```bash
python pygame_visualize.py
```

## Environment Details

### State Space
- Altitude: 0-5000 m
- Speed: 0-300 knots
- Distance to runway: 0-10000 m
- Pitch angle: -30 to +30 degrees
- Runway condition: 0.0 (dry), 0.5 (wet), 1.0 (icy)

### Action Space
- 0: Increase throttle
- 1: Decrease throttle
- 2: Increase pitch (pitch up)
- 3: Decrease pitch (pitch down)
- 4: Do nothing

### Landing Success Criteria
- Altitude: 0-50 m
- Speed: 100-200 knots
- Pitch angle: < 10 degrees
- Distance: ≤ 0 m (reached runway)

## Results

The agent learns to perform safe landings with varying success rates across different runway conditions. Training progress is logged, and checkpoints are saved periodically.

## Related Work

This project is inspired by research in:
- Graph-Enhanced Deep-Reinforcement Learning for Aircraft Landing
- Robust Auto-Landing Control Using Fuzzy Q-Learning
- Data-Efficient Deep Reinforcement Learning for UAV Attitude Control

## License

MIT License

## Author

FlightRL Project

