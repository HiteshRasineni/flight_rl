import argparse
import matplotlib.pyplot as plt
import torch

from envs.flight_env import FlightEnv
from agents.dqn_agent import DQNAgent

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()

def visualize(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'

    env = FlightEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, device=device)
    agent.load(args.checkpoint)

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False

        altitudes = []
        distances = []

        while not done:
            action = agent.act(obs, epsilon=0.0)  # greedy
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            altitudes.append(obs[0]*5000)  # de-normalize
            distances.append(obs[2]*10000)

        plt.figure(figsize=(8,4))
        plt.plot(distances, altitudes)
        plt.xlabel("Distance to Runway (m)")
        plt.ylabel("Altitude (m)")
        plt.title(f"Episode {ep} Landing Trajectory | Success: {info.get('success', False)} | Runway: {env.runway_condition}")
        plt.gca().invert_xaxis()  # distance decreases toward runway
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    args = parse_args()
    visualize(args)
