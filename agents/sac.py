# agents/sac.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from utils.replay import ReplayBuffer
from safetensors.torch import load_file

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [action_dim])

    def forward(self, obs):
        return self.net(obs)

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [action_dim])

    def forward(self, obs):
        return self.net(obs)

class SACAgent:
    def __init__(self, obs_dim, action_dim, config: dict):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.batch_size = config["batch_size"]
        self.policy_delay = config["policy_delay"]
        self.update_counter = 0

        self.actor = Actor(obs_dim, action_dim, config["hidden_sizes"]).to(DEVICE)
        self.q1 = Critic(obs_dim, action_dim, config["hidden_sizes"]).to(DEVICE)
        self.q2 = Critic(obs_dim, action_dim, config["hidden_sizes"]).to(DEVICE)
        self.q1_target = Critic(obs_dim, action_dim, config["hidden_sizes"]).to(DEVICE)
        self.q2_target = Critic(obs_dim, action_dim, config["hidden_sizes"]).to(DEVICE)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=config["lr_actor"])
        self.q_optim = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=config["lr_critic"])

        if config["target_entropy"] == "auto":
            self.target_entropy = -0.98 * np.log(1.0 / action_dim)
        else:
            self.target_entropy = float(config["target_entropy"])
        
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=config["lr_alpha"])
        
        self.replay = ReplayBuffer(config["buffer_size"])

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @staticmethod
    def _extract_obs(x):
        return x[0] if isinstance(x, (tuple, list)) else x

    def select_action(self, obs, deterministic=False):
        
        with torch.no_grad():
            obs_arr = self._extract_obs(obs)
            obs_t = torch.tensor(np.array(obs_arr), dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits = self.actor(obs_t)
            if deterministic:
                action = torch.argmax(logits, dim=1)
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()
            return action.item()

    def store_transition(self, s, a, r, ns, done):
        self.replay.push(s, a, r, ns, done)

    def update(self):
        if len(self.replay) < self.batch_size:
            return

        s, a, r, ns, d = self.replay.sample(self.batch_size)
        s = torch.tensor(s, dtype=torch.float32, device=DEVICE)
        a = torch.tensor(a, dtype=torch.int64, device=DEVICE).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        ns = torch.tensor(ns, dtype=torch.float32, device=DEVICE)
        d = torch.tensor(d, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        # --- Update Critics ---
        with torch.no_grad():
            next_logits = self.actor(ns)
            next_action_probs = F.softmax(next_logits, dim=1)
            next_log_action_probs = F.log_softmax(next_logits, dim=1)
            q1_next_target = self.q1_target(ns)
            q2_next_target = self.q2_target(ns)
            min_q_next_target = torch.min(q1_next_target, q2_next_target)
            next_value = (next_action_probs * (min_q_next_target - self.alpha * next_log_action_probs)).sum(dim=1, keepdim=True)
            q_target = r + self.gamma * (1.0 - d) * next_value

        q1_current = self.q1(s).gather(1, a)
        q2_current = self.q2(s).gather(1, a)
        q_loss = F.mse_loss(q1_current, q_target) + F.mse_loss(q2_current, q_target)
        
        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        self.update_counter += 1
        
        # --- Delayed Actor and Alpha Update ---
        if self.update_counter % self.policy_delay == 0:
            for p in self.q1.parameters(): p.requires_grad = False
            for p in self.q2.parameters(): p.requires_grad = False
            
            logits = self.actor(s)
            action_probs = F.softmax(logits, dim=1)
            log_action_probs = F.log_softmax(logits, dim=1)
            
            q1_pi = self.q1(s)
            q2_pi = self.q2(s)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (action_probs * (self.alpha.detach() * log_action_probs - min_q_pi)).sum(dim=1).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            for p in self.q1.parameters(): p.requires_grad = True
            for p in self.q2.parameters(): p.requires_grad = True

            alpha_loss = -(self.log_alpha * (log_action_probs.detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            # --- Soft Update Target Networks ---
            with torch.no_grad():
                for q1_p, q1_t_p in zip(self.q1.parameters(), self.q1_target.parameters()):
                    q1_t_p.data.mul_(1.0 - self.tau)
                    q1_t_p.data.add_(self.tau * q1_p.data)
                for q2_p, q2_t_p in zip(self.q2.parameters(), self.q2_target.parameters()):
                    q2_t_p.data.mul_(1.0 - self.tau)
                    q2_t_p.data.add_(self.tau * q2_p.data)
    
    def save(self, path):
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)

    def load(self, path):
        # --- START OF MODIFIED LOAD FUNCTION ---
        if path.endswith(".safetensors"):
            print("Loading model from .safetensors file...")
            flat_checkpoint = load_file(path, device=DEVICE)
            
            # Reconstruct the necessary state dictionaries from the flattened keys
            actor_dict = {k.split('.', 1)[1]: v for k, v in flat_checkpoint.items() if k.startswith('actor_state_dict.')}
            q1_dict = {k.split('.', 1)[1]: v for k, v in flat_checkpoint.items() if k.startswith('q1_state_dict.')}
            q2_dict = {k.split('.', 1)[1]: v for k, v in flat_checkpoint.items() if k.startswith('q2_state_dict.')}

            self.actor.load_state_dict(actor_dict)
            self.q1.load_state_dict(q1_dict)
            self.q2.load_state_dict(q2_dict)
            
            # Load single tensors like log_alpha
            if 'log_alpha' in flat_checkpoint:
                self.log_alpha = flat_checkpoint['log_alpha']

        else: # Original logic for loading .pt files
            print("Loading model from .pt file...")
            checkpoint = torch.load(path, map_location=DEVICE)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.q1.load_state_dict(checkpoint['q1_state_dict'])
            self.q2.load_state_dict(checkpoint['q2_state_dict'])
            if 'log_alpha' in checkpoint:
                self.log_alpha = checkpoint['log_alpha']
        
        # --- COMMON LOGIC FOR BOTH FORMATS ---
        # Ensure target networks and optimizers are correctly initialized after loading
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.config["lr_alpha"])
        print("Model loaded successfully.")
        # --- END OF MODIFIED LOAD FUNCTION ---