# -*- coding: utf-8 -*-
"""Implements Policy for REINFORCE algorithm.
"""
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

EPS = np.finfo(np.float32).eps.item()
SavedAction = namedtuple("SavedAction", ["log_prob", "value"])


class Reinforce(nn.Module):
    """Implements "REINFORCE" algorithm with `PyTorch`.
    
    Reference:
    https://github.com/pytorch/examples/tree/master/reinforcement_learning
    """
    def __init__(self,
                 input_nodes,
                 output_nodes,
                 gamma,
                 learning_rate,
                 optimizer=None):
        super(Reinforce, self).__init__()
        self.fc1 = nn.Linear(input_nodes, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, output_nodes)

        self.saved_log_probs = []
        self.rewards = []
        self.gamma = gamma
        if optimizer is not None:
            self.optimizer = optimizer(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.out(x)
        return F.softmax(x, dim=-1)

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
        # Calculates discounted rewards
        for r in self.rewards[::-1]:
            R = R * self.gamma + r
            returns.insert(0, R)
        # Normalizes the returns
        returns = torch.Tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + EPS)
        for log_prob, ret in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * ret)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        del self.rewards[:]
        del self.saved_log_probs[:]


class ActorCritic(nn.Module):
    """Implements "Actor-Critic" algorithm with `PyTorch`.
    
    Reference:
    https://github.com/pytorch/examples/tree/master/reinforcement_learning
    """
    def __init__(self,
                 input_nodes,
                 output_nodes,
                 gamma,
                 learning_rate,
                 optimizer=None):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_nodes, 128)
        self.fc2 = nn.Linear(128, 64)
        # Actor's layer
        self.action_head = nn.Linear(64, output_nodes)
        # Critic's layer
        self.value_head = nn.Linear(64, 1)
        self.saved_actions = []
        self.rewards = []
        self.gamma = gamma
        if optimizer is not None:
            self.optimizer = optimizer(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        # Actor chooses the action to take.
        action_prob = F.softmax(self.action_head(x), dim=-1)
        # Critic evaluates.
        state_value = self.value_head(x)
        return action_prob, state_value

    def select_action(self, observation):
        state = torch.from_numpy(observation).float().unsqueeze(0)
        probs, state_value = self(state)
        # Probabilities of actions
        actions = Categorical(probs)
        # Takes the action of the highest probability.
        action = actions.sample()
        self.saved_actions.append(
            SavedAction(actions.log_prob(action), state_value))
        return action.item()

    def update(self):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        returns = []
        # Calculates the discounted value
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        # Normalizes the returns
        returns = torch.Tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + EPS)

        for (log_prob, value), ret in zip(saved_actions, returns):
            advantage = ret - value.item()

            # Calculates actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # Calculates critic (value) loss using L1 smooth loss
            value_loss = F.smooth_l1_loss(value, torch.Tensor([ret]))
            value_losses.append(value_loss)

        self.optimizer.zero_grad()
        # Sums up all losses
        loss = (torch.stack(policy_losses).sum() +
                torch.stack(value_losses).sum())
        loss.backward()
        self.optimizer.step()

        del self.saved_actions[:]
        del self.rewards[:]
