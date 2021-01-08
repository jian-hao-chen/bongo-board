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

    AVAILABLE_TORQUE = [-1., 0., +1]

    # Use dynamics equations from the nips paper or the book.
    book_or_nips = "book"
    # timespan
    dt = 0.2

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

    def step(self, action):
        s = self.state
        torque = self.AVAILABLE_TORQUE[action]
        s_torque = np.append(s, torque)

        ns = rk4(self.__dsdt, s_torque, [0, self.dt])
        # Only care about final timestep of integration returned by integrator.
        ns = ns[-1]
        # Omit action.
        ns = ns[:4]
        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns

    def __dsdt(self, s_augmented, t):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = (m1 * lc1**2 + m2 *
              (l1**2 + lc2**2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2)
        d2 = m2 * (lc2**2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.)
        phi1 = (-m2 * l1 * lc2 * dtheta2**2 * sin(theta2) -
                2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2) +
                (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2) + phi2)

        if self.book_or_nips == "nips":
            # The following line is consistent with the description in the
            # paper.
            ddtheta2 = ((a + d2 / d1 * phi1 - phi2) /
                        (m2 * lc2**2 + I2 - d2**2 / d1))
        else:
            # The following line is consistent with the java implementation
            # and the book.
            ddtheta2 = ((a + d2 / d1 * phi1 -
                         m2 * l1 * lc2 * dtheta1**2 * sin(theta2) - phi2) /
                        (m2 * lc2**2 + I2 - d2**2 / d1))
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.)

    def reset(self):
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4, ))
        return self.__get_observation()

    def __get_observation(self):
        s = self.state
        return np.array(
            [cos(s[0]), sin(s[0]),
             cos(s[1]), sin(s[1]), s[2], s[3]])

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.

    This is a toy implementation which may be useful if you find yourself
    stranded on a system without `scipy`. Otherwise use `scipy.integrate`.
    
    Args:
        derivs: the derivative of the system and has the signature
                ``dy = derivs(yi, ti)``
        y0: initial state vector
        t: sample times
        args: additional arguments passed to the derivative function
        kwargs: additional keyword arguments passed to the derivative function

    Example 1 ::
        ## 2D system
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)

    Example 2 ::
        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivs, y0, t)
    If you have access to `scipy`, you should probably be using the
    `scipy.integrate` tools rather than this function.

    Returns:
        yout: Runge-Kutta approximation of the ODE
    """
    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t), ), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0
    for i in np.arange(len(t) - 1):
        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout


def wrap(x, minimum, maximum):
    """
    Wraps ``x`` so min <= x <= max; but unlike ``bound()`` which truncates,
    ``wrap()`` wraps x around the coordinate system defined by min, max.
    For example, min = -180, M = 180 (degrees), x = 360 --> returns 0.

    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range
    
    Returns:
        x: a scalar, wrapped
    """
    diff = maximum - minimum
    while x > maximum:
        x = x - diff
    while x < minimum:
        x = x + diff
    return x


def bound(x, minimum, maximum=None):
    """
    Either have `minimum` as scalar, so `bound(x, m, M)` which returns
    m <= x <= M *OR* have `minimum` as length 2 vector, `bound(x,m, <IGNORED>)`
    returns m[0] <= x <= m[1].

    Args:
        x: scalar
    
    Returns:
        x: scalar, bound between min (m) and Max (M)
    """
    if maximum is None:
        maximum = minimum[1]
        minimum = minimum[0]
    # bound x between min (m) and Max (M)
    return min(max(x, minimum), maximum)
