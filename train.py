# -*- coding: utf-8 -*-
"""Training script.
"""
import argparse
import numpy as np
import torch

from bongo_board import BongoBoard
from policy import Reinforce

parser = argparse.ArgumentParser(description='Bongo Board training script.')
parser.add_argument('--episodes',
                    type=int,
                    default=2000,
                    metavar='EPISODES',
                    dest='EPISODES',
                    help='max episodes (default: 1000)')
parser.add_argument('--gamma',
                    type=float,
                    default=0.99,
                    metavar='GAMMA',
                    dest='GAMMA',
                    help='discount factor (default: 0.99)')
parser.add_argument('--alpha',
                    type=float,
                    default=0.001,
                    metavar='ALPHA',
                    dest='ALPHA',
                    help='learning rate (default: 0.001)')
parser.add_argument('--seed',
                    type=int,
                    default=9527,
                    metavar='SEED',
                    dest='SEED',
                    help='random seed (default: 543)')
parser.add_argument('--no-render',
                    action='store_true',
                    dest='NO_RENDER',
                    help='set to disable render.')
args = parser.parse_args()

TARGET_REWARD = 500


def main():
    import gym
    env = BongoBoard()
    # env = gym.make("CartPole-v1")
    env.seed(args.SEED)
    torch.manual_seed(args.SEED)
    env.reset()

    input_nodes = env.observation_space.shape[0]
    output_nodes = env.action_space.n
    policy = Reinforce(input_nodes, output_nodes, args.GAMMA, args.ALPHA,
                       torch.optim.Adam)
    moving_reward = 0
    for ep in range(1, args.EPISODES + 1):
        observation, ep_reward = env.reset(), 0
        # The max steps per episodes.
        for t in range(1, 1001):
            action = policy.select_action(observation)
            observation, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            ep_reward += reward
            if not args.NO_RENDER:
                env.render()
            if done:
                break
        # Press `Esc` to early exit.
        if not args.NO_RENDER:
            if not env.viewer.isopen:
                break
        # Updates policy.
        policy.update()
        moving_reward = 0.1 * ep_reward + (1 - 0.1) * moving_reward
        print(f"Episode: {ep:4d}, " + f"Last reward: {ep_reward:4.2f}, " +
              f"Average reward: {moving_reward:4.2f}")
        if moving_reward > TARGET_REWARD:
            print(f"Done. The moving reward = {moving_reward}, " +
                  f"last episode runs {t} steps.")
            break

    env.close()


if __name__ == "__main__":
    main()