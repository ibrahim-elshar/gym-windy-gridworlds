# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import sys

class StochWindyGridWorldEnv(gym.Env):
    '''Creates the Stochastic Windy GridWorld Environment
       NOISE_CASE = 1: the noise is a scalar added to the wind tiles, i.e,
                       all wind tiles are changed by the same amount              
       NOISE_CASE = 2: the noise is a vector added to the wind tiles, i.e,
                       wind tiles are changed by different amounts.
    '''
    def __init__(self, GRID_HEIGHT=7, GRID_WIDTH=10,\
                 WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0], \
                 START_CELL = (3, 0), GOAL_CELL = (3, 7),\
                 REWARD = -1, RANGE_RANDOM_WIND=1,\
                 PROB=[1./3, 1./3, 1./3],\
                 NOISE_CASE = 1):
        self.seed()
        self.grid_height = GRID_HEIGHT
        self.grid_width = GRID_WIDTH
        self.grid_dimensions = (self.grid_height, self.grid_width)
        self.wind = np.array(WIND)
        self.realized_wind = self.wind
        self.start_cell = START_CELL
        self.goal_cell = GOAL_CELL   
        self.start_state = self.dim2to1(START_CELL)
        self.goal_state = self.dim2to1(GOAL_CELL)        
        self.reward = REWARD
        self.range_random_wind = RANGE_RANDOM_WIND
        self.w_range = np.arange(-self.range_random_wind, self.range_random_wind + 1 )
        self.probablities = PROB
        self.w_prob = dict(zip(self.w_range, self.probablities))
        self.action_space =  spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
                spaces.Discrete(self.grid_height),
                spaces.Discrete(self.grid_width)))
        self.actions = { 'U':0,   #up
                         'R':1,   #right
                         'D':2,   #down
                         'L':3 }  #left
        self.nA = len(self.actions)
        self.nS = self.dim2to1((self.grid_height-1,self.grid_width-1)) + 1
        self.num_wind_tiles = np.count_nonzero(self.wind)
        self.noise_case = NOISE_CASE
        self.P=[]
        self.all_possible_wind_values =  np.unique(self.w_range[:, None] + self.wind)
        # create transition function
        self.f = np.zeros((self.nS, self.nA, len(self.w_range)), dtype=int)
        for s in range(self.nS):
            for w in (self.w_range + self.range_random_wind): 
                # note that when w_range=[-1,0,1] then w_range + 1=[0,1,2] 
                # so we can map w values to indices by adding 1 to the value
                if s ==  self.goal_state: 
                    self.f[s,self.actions['U'], w] = self.goal_state
                    self.f[s,self.actions['R'], w] = self.goal_state                                                       
                    self.f[s,self.actions['D'], w] = self.goal_state
                    self.f[s,self.actions['L'], w] = self.goal_state                    
                else:
                    i, j = self.dim1to2(s)
                    if self.wind[j] != 0: 
                        wind = self.wind[j] + w - 1
                    else: 
                        wind = 0
                    self.f[s,self.actions['U'], w] = self.dim2to1((max(i - 1 - wind, 0), j))
                    self.f[s,self.actions['R'], w] = self.dim2to1((max(i - wind, 0),\
                                                       min(j + 1, self.grid_width - 1)))
                    self.f[s,self.actions['D'], w] = self.dim2to1((max(min(i + 1 - wind, \
                                                        self.grid_height - 1), 0), j))
                    self.f[s,self.actions['L'], w] = self.dim2to1((max(i - wind, 0),\
                                                       max(j - 1, 0)))
 
                    
        
    def dim2to1(self, cell):
        '''Transforms the 2 dim position in a grid world to 1 state'''
        return np.ravel_multi_index(cell, self.grid_dimensions)
    
    def dim1to2(self, state):
        '''Transforms the state in a grid world back to its 2 dim cell'''
        return np.unravel_index(state, self.grid_dimensions)
    
    # this function is depreciated and _virtual_step_f is used instead            
    def _virtual_step(self, state, action, force_noise=None):
        '''set up destinations for each action in each state'''
        i, j= self.dim1to2(state)
        ##############
        if force_noise is None:
            # case 1 where all wind tiles are affected by the same noise scalar, 
            # noise1 is a scalar value added to wind
            noise1 = self.np_random.choice(self.w_range, 1, self.probablities)[0] 
            # case 2  where each wind tile is affected by a different noise 
            # noise2 is a vector added to wind
            noise2 = self.np_random.choice(self.w_range, self.num_wind_tiles, self.probablities)
            noise = noise1 if self.noise_case==1 else noise2
        else:   
            noise = force_noise
        #print('noise=', noise)
        wind = np.copy(self.wind)
        wind[np.where( wind > 0 )] += noise 
        ##############
        destination = dict()
        destination[self.actions['U']] = self.dim2to1((max(i - 1 - wind[j], 0), j))
        destination[self.actions['R']] = self.dim2to1((max(i - wind[j], 0),\
                                           min(j + 1, self.grid_width - 1)))
        destination[self.actions['D']] = self.dim2to1((max(min(i + 1 - wind[j], \
                                            self.grid_height - 1), 0), j))
        destination[self.actions['L']] = self.dim2to1((max(i - wind[j], 0),\
                                               max(j - 1, 0)))

        if state ==  self.goal_state: destination[action] = self.goal_state
        if destination[action] ==  self.goal_state:
            reward = 0
            isdone = True
        else:
            reward = -1
            isdone = False
        return  destination[action], reward, isdone, wind, noise

    def _virtual_step_f(self, state, action, force_noise=None):
        '''Set up destinations for each action in each state only works with case 1
           and use the lookup table self.f. Much faster than _virtual_step.
        '''
        if force_noise is None:
            # case 1 where all wind tiles are affected by the same noise scalar, 
            # noise1 is a scalar value added to wind
            noise = self.np_random.choice(self.w_range, 1, self.probablities)[0] 
        else:   
            noise = force_noise 
        #print('noise=', noise)
        wind = np.copy(self.wind)
        wind[np.where( wind > 0 )] += noise         
        destination = self.f[state, action, noise + 1]
        if destination ==  self.goal_state:
            reward = 0
            isdone = True
        else:
            reward = -1
            isdone = False
        return  destination, reward, isdone, wind, noise
    
    def step(self, action, force_noise=None):
        """
        Parameters
        ----------
        action : 0 = Up, 1 = Right, 2 = Down, 3 = Left

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                 Agent current position in the grid.
            reward (float) :
                 Reward is -1 at every step except at goal state.
            episode_over (bool) :
                 True if the agent reaches the goal, False otherwise.
            info (dict) :
                 Contains the realized noise that is added to the wind in each 
                 step. However, official evaluations of your agent are not 
                 allowed to use this for learning.
        """
        assert self.action_space.contains(action)
        self.observation, reward, isdone, wind, noise  = self._virtual_step_f(self.observation, action, force_noise)
        self.realized_wind = wind
        return self.observation, reward, isdone, {'noise':noise}
        
    def reset(self):
        ''' resets the agent position back to the starting position'''
        self.observation = self.start_state
        return self.observation   

    def render(self, mode='human', close=False):
        ''' Renders the environment. Code borrowed and them modified 
            from https://github.com/dennybritz/reinforcement-learning'''
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = self.dim1to2(s)
            # print(self.s)
            if self.observation == s:
                output = " x "
            elif position == self.goal_cell:
                output = " T "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.grid_dimensions[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        for i in range(len(self.realized_wind)):
            output =' ' + str(self.realized_wind[i]) + ' '
            if i == 0:
                output = output.lstrip()
            if i == len(self.realized_wind) - 1:
                output = output.rstrip()
                output += "\n"
            
            outfile.write(output)
           
        outfile.write("\n")
    
    def seed(self, seed=None):
        ''' sets the seed for the envirnment'''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
         

        
