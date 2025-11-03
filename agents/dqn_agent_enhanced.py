"""
Enhanced DQN Agent with Double DQN and Dueling Architecture
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DuelingQNetwork(nn.Module):
    """Dueling DQN architecture: separates value and advantage"""
    
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        
        # Shared feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Value stream (scalar)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream (n_actions)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    
    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # This ensures identifiability
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class EnhancedDQNAgent:
    """
    Enhanced DQN with:
    - Double DQN
    - Dueling architecture
    - Gradient clipping
    """
    
    def __init__(self, obs_dim, n_actions, lr=5e-4, gamma=0.99, device='cpu', use_double=True, use_dueling=True):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.use_double = use_double
        self.use_dueling = use_dueling

        # Network architecture
        if use_dueling:
            self.q_net = DuelingQNetwork(obs_dim, n_actions).to(self.device)
            self.target_q = DuelingQNetwork(obs_dim, n_actions).to(self.device)
        else:
            # Standard Q-network
            self.q_net = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, n_actions)
            ).to(self.device)
            self.target_q = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, n_actions)
            ).to(self.device)
        
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
        """Single gradient step using Double DQN if enabled."""
        states = torch.tensor(batch['states'], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(batch['next_states'], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.int64, device=self.device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch['dones'], dtype=torch.float32, device=self.device)

        # Current Q-values
        q_values = self.q_net(states)
        q_val = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            if self.use_double:
                # Double DQN: use online network to select action, target network to evaluate
                next_q_online = self.q_net(next_states)
                next_actions = next_q_online.argmax(dim=1)
                next_q_target = self.target_q(next_states)
                next_q_max = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q = self.target_q(next_states)
                next_q_max, _ = next_q.max(dim=1)
            
            target = rewards + (1.0 - dones) * self.gamma * next_q_max

        loss = self.loss_fn(q_val, target)

        self.opt.zero_grad()
        loss.backward()
        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.opt.step()

        return loss.item()

    def sync_target(self):
        """Copy weights from online network to target network"""
        self.target_q.load_state_dict(self.q_net.state_dict())

    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'q_state': self.q_net.state_dict(),
            'target_state': self.target_q.state_dict(),
            'opt_state': self.opt.state_dict(),
            'config': {
                'obs_dim': self.obs_dim,
                'n_actions': self.n_actions,
                'use_double': self.use_double,
                'use_dueling': self.use_dueling
            }
        }, path)

    def load(self, path: str, map_location=None):
        """Load agent state"""
        d = torch.load(path, map_location=map_location)
        self.q_net.load_state_dict(d['q_state'])
        self.target_q.load_state_dict(d.get('target_state', d['q_state']))
        try:
            self.opt.load_state_dict(d.get('opt_state', self.opt.state_dict()))
        except Exception:
            pass

