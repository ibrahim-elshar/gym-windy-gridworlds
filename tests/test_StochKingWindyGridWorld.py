# -*- coding: utf-8 -*-
import gym
import gym_windy_gridworlds


env = gym.make('StochKingWindyGridWorld-v0')
env.seed(10703)
env.reset()

done = False

while not done:
    action = env.action_space.sample()
    print(action)
    observation, reward, done, info = env.step(action)
    print(observation, reward, done, info)
    