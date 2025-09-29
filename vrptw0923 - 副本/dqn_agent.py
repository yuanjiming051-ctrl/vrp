# File: dqn_agent.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """经验回放缓冲区，存储 (state, action, reward, next_state, done) 元组"""

    def __init__(self, buffer_size=int(1e5), batch_size=8, seed=0):
        random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """加入一个 transition"""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """随机采样 batch"""
        batch = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(
            np.vstack([x[0] for x in batch if x is not None])
        ).float()
        actions = torch.from_numpy(
            np.vstack([x[1] for x in batch if x is not None])
        ).long()
        rewards = torch.from_numpy(
            np.vstack([x[2] for x in batch if x is not None])
        ).float()
        next_states = torch.from_numpy(
            np.vstack([x[3] for x in batch if x is not None])
        ).float()
        dones = torch.from_numpy(
            np.vstack([x[4] for x in batch if x is not None]).astype(np.uint8)
        ).float()
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """
    DQN Agent：维护 local & target Q 网络、经验回放，提供 act() 和 step() 接口。
    """

    def __init__(self, state_size, action_size, seed=0,
                 buffer_size=int(1e5), batch_size=16, lr=1e-4, gamma=0.995,
                 tau=5e-4, update_every=2, verbose: bool = False):
        self.state_size = state_size
        self.action_size = action_size
        random.seed(seed)
        self.verbose = verbose

        # Q 网络（local & target），改为多层网络结构
        self.qnetwork_local = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(device)
        
        self.qnetwork_target = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(device)
        
        # 初始化网络权重，确保Q值在合理范围内
        self._init_weights()
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr, weight_decay=1e-4)

        # 超参
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every

        # 经验回放
        self.memory = ReplayBuffer(buffer_size=int(1e5),
                                   batch_size=16,  # 与初始化参数保持一致
                                   seed=seed)
        self.t_step = 0

    def _init_weights(self):
        """初始化网络权重，防止Q值偏向正值或负值"""
        for module in self.qnetwork_local.modules():
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                # 最后一层（输出层）使用正偏置，确保Q值偏向正值
                if module == list(self.qnetwork_local.modules())[-1]:
                    nn.init.constant_(module.bias, 0.5)  # 正偏置
                else:
                    nn.init.constant_(module.bias, 0.0)
        
        # 同样初始化目标网络
        for module in self.qnetwork_target.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module == list(self.qnetwork_target.modules())[-1]:
                    nn.init.constant_(module.bias, 0.5)
                else:
                    nn.init.constant_(module.bias, 0.0)

    def step(self, state, action, reward, next_state, done):
        """存样本并每 update_every 步 learn 一次"""
        self.memory.add(state, action, reward, next_state, done)
        if self.verbose:
            print(f"[DQNAgent.step] buffer={len(self.memory)}/{self.memory.batch_size}")
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.memory.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, eps=0.0):
        """ε-greedy 选动作"""
        state = state.to(next(self.qnetwork_local.parameters()).device)
        with torch.no_grad():
            q_values = self.qnetwork_local(state).cpu().numpy()
        if random.random() > eps:
            return np.argmax(q_values)
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """用一批经验训练 local 网络，并软更新 target 网络"""
        states, actions, rewards, next_states, dones = experiences
        states      = states.to(device)
        actions     = actions.to(device)
        rewards     = rewards.to(device)
        next_states = next_states.to(device)
        dones       = dones.to(device)

        # Q_target = r + γ * max_a' Q_target(next, a') * (1 - done)
        Q_targets_next = self.qnetwork_target(next_states) \
                               .detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Q_expected = Q_local(states, actions)
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        if self.verbose:
            print(f"[DQNAgent.learn] loss={loss.item():.6f}")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        """软更新 target: θ_target = τθ_local + (1-τ)θ_target"""
        for t, l in zip(target_model.parameters(), local_model.parameters()):
            t.data.copy_(self.tau * l.data + (1.0 - self.tau) * t.data)
