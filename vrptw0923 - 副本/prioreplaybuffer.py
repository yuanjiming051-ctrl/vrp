# File: prioreplaybuffer.py

import random
import numpy as np
import torch
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """
    简单的经验回放缓冲区，用于存储 (state, action, reward, next_state, done) 并采样小批量。
    """

    def __init__(self, buffer_size=int(1e5), batch_size=64, seed=0):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.vstack([e[0] for e in experiences if e is not None])
        actions = np.vstack([e[1] for e in experiences if e is not None])
        rewards = np.vstack([e[2] for e in experiences if e is not None])
        next_states = np.vstack([e[3] for e in experiences if e is not None])
        dones = np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)

        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
