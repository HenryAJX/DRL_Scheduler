# agents/d3qn.py
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import DQN
from utils.replay import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.feature = nn.Sequential(*layers)

        self.value = nn.Sequential(nn.Linear(last, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv = nn.Sequential(nn.Linear(last, 128), nn.ReLU(), nn.Linear(128, output_dim))

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.adv(f)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

class D3QNAgent:
    def __init__(self, obs_dim, action_dim, config: dict = None):
        cfg = config if config is not None else DQN
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = cfg

        self.policy_net = DuelingDQN(obs_dim, action_dim, cfg["hidden_sizes"]).to(DEVICE)
        self.target_net = DuelingDQN(obs_dim, action_dim, cfg["hidden_sizes"]).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optim = optim.Adam(self.policy_net.parameters(), lr=cfg["lr"])
        self.gamma = cfg["gamma"]
        self.batch_size = cfg["batch_size"]
        self.replay = ReplayBuffer(cfg["buffer_size"])
        
        self.epsilon_start = cfg["epsilon_start"]
        self.epsilon_end = cfg["epsilon_end"]
        self.epsilon_decay_steps = cfg["epsilon_decay_steps"]
        self.total_steps = 0
        self.target_update_freq = cfg["target_update_freq"]
    
    @staticmethod
    def _extract_obs(x):
        """Extracts the observation array from a potential (obs, info) tuple."""
        return x[0] if isinstance(x, (tuple, list)) else x

    @staticmethod
    def _normalize_done(done):
        """Converts a (terminated, truncated) tuple into a single boolean."""
        return bool(any(done)) if isinstance(done, (tuple, list)) else bool(done)

    def select_action(self, obs):
        self.total_steps += 1
        eps = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
              math.exp(-1.0 * self.total_steps / max(1, self.epsilon_decay_steps))

        obs_arr = self._extract_obs(obs)

        if np.random.rand() < eps:
            return np.random.randint(0, self.action_dim)

        obs_t = torch.tensor(np.array(obs_arr), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        self.policy_net.eval()
        with torch.no_grad():
            q = self.policy_net(obs_t)
        self.policy_net.train()
        return int(q.argmax(dim=1).item())

    def store_transition(self, s, a, r, ns, done):
        s_arr = self._extract_obs(s)
        ns_arr = self._extract_obs(ns)
        done_flag = self._normalize_done(done)
        self.replay.push(s_arr, a, r, ns_arr, done_flag)

    def update(self):
        if len(self.replay) < self.batch_size:
            return 0.0

        s, a, r, ns, d = self.replay.sample(self.batch_size)
        s = torch.tensor(np.array(s), dtype=torch.float32, device=DEVICE)
        a = torch.tensor(np.array(a), dtype=torch.int64, device=DEVICE).unsqueeze(1)
        r = torch.tensor(np.array(r), dtype=torch.float32, device=DEVICE).unsqueeze(1)
        ns = torch.tensor(np.array(ns), dtype=torch.float32, device=DEVICE)
        d = torch.tensor(np.array(d), dtype=torch.float32, device=DEVICE).unsqueeze(1)

        with torch.no_grad():
            next_q_policy = self.policy_net(ns)
            next_actions = next_q_policy.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(ns).gather(1, next_actions)
            target = r + self.gamma * (1.0 - d) * next_q_target

        q_vals = self.policy_net(s).gather(1, a)
        loss = nn.MSELoss()(q_vals, target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        st = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(st)
        self.target_net.load_state_dict(self.policy_net.state_dict())