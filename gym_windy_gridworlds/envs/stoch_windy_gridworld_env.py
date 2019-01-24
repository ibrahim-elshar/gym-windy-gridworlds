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
                 REWARD = -1, RANGE_RANDOM_WIND=2,\
                 PROB=[0.35, 0.1, 0.1, 0.1, 0.35],\
                 NOISE_CASE = 1,
                 SIMULATOR_SEED = 3323,
                 GAMMA = 0.9):
        self.prng_simulator = np.random.RandomState(SIMULATOR_SEED) #Pseudorandom number generator
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
        self.probabilities = PROB
        self.w_prob = dict(zip(self.w_range, self.probabilities))
        self.action_space =  spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
                spaces.Discrete(self.grid_height),
                spaces.Discrete(self.grid_width)))
        self.seed()
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
#                    print(i,j)
                    
                    if self.wind[j] != 0: 
                        wind = self.wind[j] + w - self.range_random_wind
#                        print('wind=',wind)
                    else: 
                        wind = 0
                    self.f[s,self.actions['U'], w] = self.dim2to1((max(i - 1 - wind, 0), j))
                    self.f[s,self.actions['R'], w] = self.dim2to1((min(max(i - wind, 0),self.grid_height - 1),\
                                                       min(j + 1, self.grid_width - 1)))
                    self.f[s,self.actions['D'], w] = self.dim2to1((max(min(i + 1 - wind, \
                                                        self.grid_height - 1), 0), j))
                    self.f[s,self.actions['L'], w] = self.dim2to1((min(max(i - wind, 0),self.grid_height - 1),\
                                                       max(j - 1, 0)))
        # create transition probabilities           
        self.P=np.zeros((self.nS,self.nA,self.nS)) 
        for s in range(self.nS):
            for a in range(self.nA):
                self.P[s,a][np.unique(self.f[s,a,:])]= \
                [list(self.f[s,a,:]).count(value)/float(len(self.f[s,a,:]))\
                      for value in np.unique(self.f[s,a,:])]
        # absorption formulation gamma
        self.gamma = GAMMA
        
        # creates a table that returns probabilities, next states, rewards, isdone (0 False, 1 True)
        self.trans=np.empty((self.nS,self.nA), dtype=object)
        for s in range(self.nS):
            for a in range(self.nA):
                self.trans[s,a] = np.vstack((self.P[s,a,np.where(self.P[s,a]!=0)][0],\
                          np.where(self.P[s,a]!=0)[0],self.reward_func(s, a), np.where(self.P[s,a]!=0)[0]==self.goal_state)).T
        
    def reward_func(self,state, action):
        next_states = np.unique(self.f[state, action, :])
        reward = np.ones(len(next_states)) * -1
        reward[np.where(next_states == self.goal_state)] = 0
#        if state==self.goal_state: 
#            reward = np.zeros(len(next_states))
        return reward   
    
    def f(self, s, a, w):
        return self.f[s, a, w + self.range_random_wind]
               
    def create_absorption_MDP_P(self):
        '''TODO'''
        self.P_new=np.zeros((self.nS+1,self.nA,self.nS+1))
        self.P_new[:,:,self.nS] = 1- self.gamma
        self.P_new[0:self.nS,:,0:self.nS] = self.gamma * self.P
        self.P_new[self.nS,:,self.nS] = 1
    
    def _virtual_step_absorb(self, s,a,force_noise=None):
        '''TODO'''
        noise = force_noise if force_noise is not None else self.np_random.choice(self.w_range, 1, p=self.probabilities)[0] 
        wind = np.copy(self.wind)
        wind[np.where( wind > 0 )] += noise  
        if s==self.nS:
            return  self.nS, 0, True, wind, noise        
        P=np.zeros((self.nS+1,self.nA,self.nS+1))
        if force_noise is None:
            newS = self.f[s,a,noise + self.range_random_wind]
#            print(newS)
            # P(s' | w) = 1_{s'=f(s,a,w)} x Gamma + 1_{s'=new_absorb_state} x 1- Gamma
            P[s,a,newS]= self.gamma 
            P[s,a,self.nS]= 1.0-self.gamma 
#            print(self.gamma)
#            print(1-self.gamma)
#            print(P[s,a,self.nS])
#            print(np.nonzero(P[s,a,:]))
            prob = P[s,a,np.nonzero(P[s,a,:])][0].tolist()
#            print(prob)
            destination = self.np_random.choice(np.append(newS,self.nS), 1, p=prob)[0]
            
#            prob = self.P_new[s,a,np.nonzero(self.P_new[s,a,:])][0].tolist()
#            destination = self.np_random.choice(np.append(np.unique(self.f[s,a,:]),self.nS), 1,\
#                                           p=prob)[0] # TODO check unique here makes the array sorted
           
            if destination ==  self.goal_state:
                reward = 0
                isdone = False
            elif destination ==  self.nS:  
                reward = 0
                isdone = True
            else:
                reward = -1
                isdone = False
            return  destination, reward, isdone, wind, noise
        else: 
#            noise = force_noise 
#            newS = self.f[s,a,noise + self.range_random_wind]
#            P[s,a,newS]= self.gamma 
#            P[s,a,self.nS]= 1-self.gamma 
#            prob = P[s,a,np.nonzero(P[s,a,:])][0].tolist()
#            destination = self.np_random.choice(np.append(newS,self.nS), 1,\
#                                           p=prob)[0]
            return self._virtual_step_f( s, a, force_noise=noise)
        
    def simulate_sample_path(self):
        '''TODO'''
        tau = self.prng_simulator.geometric(p=1-self.gamma, size=1)[0]   
        sample_path = self.prng_simulator.choice(self.w_range, tau, p=self.probabilities)
        return sample_path
        
    def step_absorb(self, action, force_noise=None):
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
        self.observation, reward, isdone, wind, noise  = self._virtual_step_absorb(self.observation, action, force_noise)
        self.realized_wind = wind
        return self.observation, reward, isdone, {'noise':noise}    
                
            
    def dim2to1(self, cell):
        '''Transforms the 2 dim position in a grid world to 1 state'''
        return np.ravel_multi_index(cell, self.grid_dimensions)
    
    def dim1to2(self, state):
        '''Transforms the state in a grid world back to its 2 dim cell'''
        return np.unravel_index(state, self.grid_dimensions)
    
    # this function is depreciated and _virtual_step_f is used instead            
#    def _virtual_step(self, state, action, force_noise=None):
#        '''set up destinations for each action in each state'''
#        i, j= self.dim1to2(state)
#        ##############
#        if force_noise is None:
#            # case 1 where all wind tiles are affected by the same noise scalar, 
#            # noise1 is a scalar value added to wind
#            noise1 = self.np_random.choice(self.w_range, 1, p=self.probabilities)[0] 
#            # case 2  where each wind tile is affected by a different noise 
#            # noise2 is a vector added to wind
#            noise2 = self.np_random.choice(self.w_range, self.num_wind_tiles, p=self.probabilities)
#            noise = noise1 if self.noise_case==1 else noise2
#        else:   
#            noise = force_noise
#        #print('noise=', noise)
#        wind = np.copy(self.wind)
#        wind[np.where( wind > 0 )] += noise 
#        ##############
#        destination = dict()
#        destination[self.actions['U']] = self.dim2to1((max(i - 1 - wind[j], 0), j))
#        destination[self.actions['R']] = self.dim2to1((min(max(i - wind, 0),self.grid_height - 1),\
#                                                       min(j + 1, self.grid_width - 1)))
#        destination[self.actions['D']] = self.dim2to1((max(min(i + 1 - wind[j], \
#                                            self.grid_height - 1), 0), j))
#        destination[self.actions['L']] = self.dim2to1((min(max(i - wind, 0),self.grid_height - 1),\
#                                                       max(j - 1, 0)))
#
#        if state ==  self.goal_state: destination[action] = self.goal_state
#        if destination[action] ==  self.goal_state:
#            reward = 0
#            isdone = True
#        else:
#            reward = -1
#            isdone = False
#        return  destination[action], reward, isdone, wind, noise

    def _virtual_step_f(self, state, action, force_noise=None):
        '''Set up destinations for each action in each state only works with case 1
           and use the lookup table self.f. Much faster than _virtual_step.
        '''
        if force_noise is None:
            # case 1 where all wind tiles are affected by the same noise scalar, 
            # noise1 is a scalar value added to wind
            noise = self.np_random.choice(self.w_range, 1, p=self.probabilities)[0] 
        else:   
            noise = force_noise 
        #print('noise=', noise)
        wind = np.copy(self.wind)
        wind[np.where( wind > 0 )] += noise         
        destination = self.f[state, action, noise + self.range_random_wind]
        #if destination ==  self.goal_state:
        if state ==self.goal_state and destination ==  self.goal_state:
            reward = 0 ########################################### 0 before
            isdone = True
        elif state !=self.goal_state and destination ==  self.goal_state:
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
        self.realized_wind = self.wind
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
         

        
