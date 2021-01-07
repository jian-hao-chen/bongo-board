# -*- coding: utf-8 -*-
"""Implements a Bongo Board environment.
"""
import gym
from gym.envs.classic_control.acrobot import AcrobotEnv


class BongoBoard(gym.Env):
    """
    Description:
        An assignment from NTNU Reinforcement Learning class.
        Assignment description page: https://reurl.cc/mq6291
    
    Source:
        This environment is modified from "Acrobot-v1" and "CartPole-v1".
    
    Observation:
        Type: Box(6)
        Num   Observation            Min    Max
        0     cos(theta)
        1     sin(theta)
        2     cos(alpha)
        3     sin(alpha)
        4     Derivative of theta
        5     Derivative of alpha

    Action:
        Type: Discrete(3)
        Num   Action
        0     Torque = +1
        1     Torque =  0
        2     Torque = -1

    Reference:
        Acrobot-v1: https://gym.openai.com/envs/Acrobot-v1/
        CartPole-v1: https://gym.openai.com/envs/CartPole-v1/
    """
    def __init__(self):
        pass


def main():
    pass


if __name__ == '__main__':
    main()