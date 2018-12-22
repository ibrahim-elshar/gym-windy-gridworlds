# -*- coding: utf-8 -*-
import gym
import gym_windy_gridworlds


env = gym.make('KingWindyGridWorld-v0')

env.reset()

done = False

while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(observation, reward, done, info)
    