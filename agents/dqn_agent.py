import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        return self.model(x)

class DQNAgent:
    def __init__(self, obs_dim, n_actions, lr=5e-4, gamma=0.99, device='cpu'):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma

        self.q_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_q = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_q.load_state_dict(self.q_net.state_dict())

        self.opt = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, obs, epsilon=0.0):
        """Epsilon-greedy action. obs: numpy array (obs_dim,)"""
        if np.random.rand() < epsilon:
            return int(np.random.randint(0, self.n_actions))
        self.q_net.eval()
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_net(t)
            action = int(q.argmax(dim=1).cpu().numpy()[0])
        self.q_net.train()
        return action

    def update(self, batch, batch_size):
        """Single gradient step using a sampled batch (dict of numpy arrays)."""
        states = torch.tensor(batch['states'], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(batch['next_states'], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.int64, device=self.device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch['dones'], dtype=torch.float32, device=self.device)

        # Current Q-values
        q_values = self.q_net(states)
        q_val = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN could be used; here simple target network)
        with torch.no_grad():
            next_q = self.target_q(next_states)
            next_q_max, _ = next_q.max(dim=1)
            target = rewards + (1.0 - dones) * self.gamma * next_q_max

        loss = self.loss_fn(q_val, target)

        self.opt.zero_grad()
        loss.backward()
        # optional gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.opt.step()

        return loss.item()

    def sync_target(self):
        self.target_q.load_state_dict(self.q_net.state_dict())

    def save(self, path: str):
        torch.save({
            'q_state': self.q_net.state_dict(),
            'target_state': self.target_q.state_dict(),
            'opt_state': self.opt.state_dict(),
        }, path)

    def load(self, path: str, map_location=None):
        d = torch.load(path, map_location=map_location)
        self.q_net.load_state_dict(d['q_state'])
        self.target_q.load_state_dict(d.get('target_state', d['q_state']))
        try:
            self.opt.load_state_dict(d.get('opt_state', self.opt.state_dict()))
        except Exception:
            pass
