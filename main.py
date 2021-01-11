# -*- coding: utf-8 -*-
"""
"""
from bongo_board import BongoBoard


def main():
    env = BongoBoard()
    env.reset()
    env.render()
    print(env.state)
    import time
    time.sleep(5)
    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            print(f"Episode finished after {None} steps.")
            break
        if not env.viewer.isopen:
            break
    env.close()


if __name__ == '__main__':
    main()