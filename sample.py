# -*- coding: utf-8 -*-
import gym
import matplotlib.pyplot as plt

from bongo_board import BongoBoard


def main():
    env = BongoBoard()
    ep_rewards = []
    moving_rewards = []
    moving_reward = 0
    for ep in range(1, 1001):
        observation, ep_reward = env.reset(), 0
        for t in range(1, 1001):
            action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                break
        moving_reward = 0.05 * ep_reward + (1 - 0.05) * moving_reward
        ep_rewards.append(ep_reward)
        moving_rewards.append(moving_reward)
    env.close()

    fig = plt.figure(dpi=144)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ep_rewards, label="Episode Reward")
    ax.plot(moving_rewards, label=" Moving Reward")
    ax.set_title("Random Results")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_ylim(0, 70)
    ax.legend()
    plt.savefig(f"img/random_sample.png")



if __name__ == "__main__":
    main()