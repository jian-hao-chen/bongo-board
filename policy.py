# -*- coding: utf-8 -*-
"""Implements Policy for REINFORCE algorithm.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(nn.Module):
    """A neural network implemented with `PyTorch` as Policy.
    
    Reference:
    https://github.com/pytorch/examples/tree/master/reinforcement_learning
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(6, 16)
        self.fc2 = nn.Linear(16, 3)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

    def select_action(self, observation):
        state = torch.from_numpy(observation)
        probs = self(state)
        actions = Categorical(probs)
        action = actions.sample()
        self.saved_log_probs.append(actions.log_prob(action))
        return action.item()