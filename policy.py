# -*- coding: utf-8 -*-
"""Implements Policy for REINFORCE algorithm.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

EPS = np.finfo(np.float32).eps.item()


class Policy(nn.Module):
    """A neural network implemented with `PyTorch` as Policy.
    
    Reference:
    https://github.com/pytorch/examples/tree/master/reinforcement_learning
    """
    def __init__(self, gamma, learning_rate, optimizer=None):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []
        self.gamma = gamma
        if optimizer is not None:
            self.optimizer = optimizer(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def select_action(self, observation):
        state = torch.from_numpy(observation).float().unsqueeze(0)
        probs = self(state)
        actions = Categorical(probs)
        action = actions.sample()
        self.saved_log_probs.append(actions.log_prob(action))
        return action.item()

    def update(self):
        R = 0
        policy_loss = []
        returns = []
        # Computes discounted rewards
        for r in self.rewards[::-1]:
            R = R * self.gamma + r
            returns.insert(0, R)
        returns = torch.Tensor(returns)
        # Normalizes the returns
        returns = (returns - returns.mean()) / (returns.std() + EPS)
        for log_prob, ret in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * ret)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]
