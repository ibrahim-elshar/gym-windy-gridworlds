# -*- coding: utf-8 -*-
import gym
import gym_windy_gridworlds
import numpy as np

env = gym.make('StochWindyGridWorld-v0')
env.seed(10703)
env.reset()

done = False
sample_path = []
while not done:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action, force_noise=None)
    sample_path.append(info)
    print(observation, reward, done, info)
    