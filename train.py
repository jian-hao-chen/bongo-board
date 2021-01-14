# -*- coding: utf-8 -*-
"""Training script.
"""
import argparse
import numpy as np
import torch

from bongo_board import BongoBoard
from policy import Reinforce, ActorCritic

parser = argparse.ArgumentParser(description='Bongo Board training script.')
parser.add_argument('--episodes',
                    type=int,
                    default=5000,
                    metavar='EPISODES',
                    dest='EPISODES',
                    help='max episodes (default: 5000)')
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
parser.add_argument('--log-interval',
                    type=int,
                    default=100,
                    dest='LOG_INTERVAL',
                    help='number of episodes for log interval')
parser.add_argument('--no-render',
                    action='store_true',
                    dest='NO_RENDER',
                    help='set to disable render.')
parser.add_argument('--a2c',
                    action='store_true',
                    dest='A2C',
                    help='set to use "Actor-Critic" as policy.')
args = parser.parse_args()


def main():
    import gym
    ep_rewards = []
    moving_rewards = []
    max_reward = 10

    env = BongoBoard()
    env.seed(args.SEED)
    torch.manual_seed(args.SEED)
    env.reset()

    input_nodes = env.observation_space.shape[0]
    output_nodes = env.action_space.n
    optimizer = torch.optim.RMSprop
    if args.A2C:
        policy = ActorCritic(input_nodes, output_nodes, args.GAMMA, args.ALPHA,
                             optimizer)
    else:
        policy = Reinforce(input_nodes, output_nodes, args.GAMMA, args.ALPHA,
                           optimizer)
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
        moving_reward = 0.05 * ep_reward + (1 - 0.05) * moving_reward
        if ep % args.LOG_INTERVAL == 0:
            print(f"Episode: {ep:4d}, " + f"Last reward: {ep_reward:4.2f}, " +
                  f"Average reward: {moving_reward:4.2f}")

        # Saves model.
        if ep_reward > max_reward:
            model_name = "a2c" if args.A2C else "reinforce"
            torch.save(policy.state_dict(),
                       f"model/{model_name}_{int(ep_reward)}.pth")
            max_moving_reward = moving_reward

        # Stores training history.
        ep_rewards.append(ep_reward)
        moving_rewards.append(moving_reward)

    env.close()

    # Plots results.
    import matplotlib.pyplot as plt
    fig = plt.figure(dpi=144)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ep_rewards, label="Episode Reward")
    ax.plot(moving_rewards, label=" Moving Reward")
    ax.set_title("Training Results")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_ylim(0, 70)
    ax.legend()
    plt.savefig(f"img/{model_name}_{args.GAMMA}_{args.ALPHA}.png")
    # plt.show()


if __name__ == "__main__":
    main()