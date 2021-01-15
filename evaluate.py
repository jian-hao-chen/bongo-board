# -*- coding: utf-8 -*-
"""Evaluates the environment by trained model.
"""
import argparse
import time

import torch

from bongo_board import BongoBoard
from policy import ActorCritic, Reinforce

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="model path")
parser.add_argument("--a2c",
                    action="store_true",
                    help="set to load an Actor-Critic model.")
args = parser.parse_args()


def main():
    env = BongoBoard()
    input_nodes = env.observation_space.shape[0]
    output_nodes = env.action_space.n
    if args.a2c:
        policy = ActorCritic(input_nodes, output_nodes)
    else:
        policy = Reinforce(input_nodes, output_nodes)
    policy.load_state_dict(torch.load(args.model))

    while True:
        observation = env.reset()
        for _ in range(1, 1001):
            action = policy.select_action(observation)
            observation, reward, done, _ = env.step(action)
            env.render()
            time.sleep(1 / 30)
            if done:
                break
        if not env.viewer.isopen:
            break


if __name__ == "__main__":
    main()