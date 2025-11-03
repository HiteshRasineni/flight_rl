import argparse
import os
import numpy as np
import torch
import pygame

from envs.flight_env import FlightEnv
from agents.dqn_agent import DQNAgent
from agents.replay_buffer import ReplayBuffer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=10000)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--buffer_size', type=int, default=50000)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--eps_start', type=float, default=1.0)
    p.add_argument('--eps_end', type=float, default=0.05)
    p.add_argument('--eps_decay_steps', type=int, default=100000)
    p.add_argument('--target_update', type=int, default=500)
    p.add_argument('--save_dir', type=str, default='./checkpoints')
    p.add_argument('--save_every', type=int, default=500)
    p.add_argument('--report_every', type=int, default=10)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def train(args):
    # Device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Environment and Agent
    env = FlightEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNAgent(obs_dim, n_actions, lr=args.lr, gamma=0.99, device=device)
    buffer = ReplayBuffer(args.buffer_size, (obs_dim,))

    os.makedirs(args.save_dir, exist_ok=True)

    epsilon = args.eps_start
    eps_decay = (args.eps_start - args.eps_end) / max(1, args.eps_decay_steps)
    total_steps = 0
    best_ep_return = -1e9

    # Track success rates by runway type
    success_counts = {0.0: 0, 0.5: 0, 1.0: 0}
    total_counts = {0.0: 0, 0.5: 0, 1.0: 0}

    # --- Pygame Setup ---
    pygame.init()
    WIDTH, HEIGHT = 800, 400
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("FlightRL Takeoff & Landing")
    clock = pygame.time.Clock()

    # Plane image (triangle)
    plane_img = pygame.Surface((40, 20), pygame.SRCALPHA)
    pygame.draw.polygon(plane_img, (255, 0, 0), [(0, 0), (40, 10), (0, 20)])

    print(f"Training DQN on FlightEnv | device={device} | episodes={args.episodes}")

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        ep_return = 0.0
        ep_steps = 0
        runway_type = env.runway_condition

        done = False
        while not done:
            # Handle quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Select action
            action = agent.act(obs, epsilon=epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done_flag = bool(terminated or truncated)

            # Store transition
            buffer.push(obs, action, reward, next_obs, done_flag)
            obs = next_obs
            ep_return += reward
            ep_steps += 1
            total_steps += 1

            # Learning step
            if len(buffer) >= args.batch_size:
                batch = buffer.sample(args.batch_size)
                agent.update(batch, args.batch_size)

            # Target network update
            if total_steps % args.target_update == 0:
                agent.sync_target()

            # Decay epsilon
            if epsilon > args.eps_end:
                epsilon -= eps_decay
                epsilon = max(epsilon, args.eps_end)

            # --- Pygame Rendering ---
            screen.fill((135, 206, 235))  # Sky
            pygame.draw.rect(screen, (50, 50, 50), (0, HEIGHT - 50, WIDTH, 50))  # Runway

            # Map env coordinates to screen
            x = int((env.distance / env.observation_space.high[2]) * WIDTH)
            y = int(HEIGHT - 50 - (env.altitude / env.observation_space.high[0]) * (HEIGHT - 50))
            rotated_plane = pygame.transform.rotate(plane_img, -env.angle)
            screen.blit(rotated_plane, (x, y))
            pygame.display.flip()
            clock.tick(30)

            if done_flag:
                # Show landing/crash for a few frames
                for _ in range(30):
                    screen.fill((135, 206, 235))
                    pygame.draw.rect(screen, (50, 50, 50), (0, HEIGHT - 50, WIDTH, 50))
                    screen.blit(rotated_plane, (x, y))
                    pygame.display.flip()
                    clock.tick(30)
                break

        # Update runway success stats
        total_counts[runway_type] += 1
        if info.get("success", False):
            success_counts[runway_type] += 1

        # Report
        if ep % args.report_every == 0:
            success_rates = {k: (success_counts[k] / total_counts[k] if total_counts[k] > 0 else 0.0) for k in success_counts}
            print(f"[Ep {ep}/{args.episodes}] Return: {ep_return:.2f} | Steps: {ep_steps} | "
                  f"Epsilon: {epsilon:.3f} | TotalSteps: {total_steps} | Success Rates: {success_rates}")

        # Save best model
        if ep_return > best_ep_return:
            best_ep_return = ep_return
            agent.save(os.path.join(args.save_dir, 'dqn_best.pt'))

        if ep % args.save_every == 0:
            agent.save(os.path.join(args.save_dir, f'dqn_ep{ep}.pt'))

    # Final save
    agent.save(os.path.join(args.save_dir, 'dqn_final.pt'))
    print("Training finished. Models saved to:", args.save_dir)

    # Final success rates
    success_rates = {k: (success_counts[k] / total_counts[k] if total_counts[k] > 0 else 0.0) for k in success_counts}
    print("Final Landing Success Rates by Runway Type:", success_rates)

    pygame.quit()

if __name__ == "__main__":
    args = parse_args()
    train(args)
