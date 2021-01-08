# -*- coding: utf-8 -*-
"""Implements a Bongo Board environment.
"""
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from numpy import cos, sin, pi


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
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 15
    }

    # TODO(jhchen): 需要將常數修改為符合作業的規範
    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.  #: moments of inertia for both links

    MAX_VEL_1 = 4 * pi
    MAX_VEL_2 = 9 * pi

    AVAIL_TORQUE = [-1., 0., +1]

    def __init__(self):
        self.viewer = None
        # TODO(jhchen): 需要確定數值範圍
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2],
                        dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.Discrete(3)
        self.state = None
        self.seed()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        


def main():
    pass


if __name__ == '__main__':
    main()