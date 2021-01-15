# -*- coding: utf-8 -*-
"""Evaluates the environment by trained model.
"""
import torch

from bongo_board import BongoBoard
from policy import ActorCritic

def main():
    env = BongoBoard()
    input_nodes = env.observation_space.shape[0]
    output_nodes = env.action_space.n
    policy = ActorCritic(input_nodes, output_nodes)
    policy.load_state_dict(torch.load("model/a2c_53.pth"))

    while True:
        observation = env.reset()
        for _ in range(1, 1001):
            action = policy.select_action(observation)
            observation, reward, done, _ = env.step(action)
            env.render()
            if done:
                break
        if not env.viewer.isopen:
            break

if __name__ == "__main__":
    main()