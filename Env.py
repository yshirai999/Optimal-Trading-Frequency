import numpy as np
from numpy.random import PCG64DXSM, Generator
import scipy as sp
from Pricers import BSprice, BGprice
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from typing import TYPE_CHECKING, Optional

class LocalVol(gym.Env):

    def __init__(self, Dynamics: str = 'BS', start_time: float = 0, T: float = 5, dT: float = 1/252, r: float = 0, mu: list[float] = [0.08,-0.02], sigma: list[float] = [0.05, 0.5]):
        self.start_time = start_time
        self.T = T # time horizon
        self.dT = dT # time step
        self.M = int(T/dT) # number of time steps
        self.tradingtimes = np.ones(self.M) # trading times
        self.r = r # risk-free rate
        self.P = [[0.8,0.05,0.1,0.05],[0.7,0.05,0.15,0.1],[0.5,0.2,0.2,0.1],[0.3,0.2,0.2,0.3]] # transition matrix
        
        self.Dynamics = Dynamics # Black Scholes with regime switching by default - other dynamics need to be implemented
        self.S0 = 100 # initial stock price
        self.ts = [] # time series of stock prices
        self.time = 0 # current time
        self.terminated = False # whether the episode is terminated
        self.truncated = False # whether the episode is truncated
        self.reward = 0 # current reward
        self.info = {} # additional information
        self.sigma = sigma # volatility
        self.mu = mu # drift

        self.Pi = [] # Portfolio value

        self.action_space = Discrete(self.M, start=dT, seed = 42) # action space is the trading frequency
        self.observation_space = Discrete(4, start=0, seed=42)  # {0, 1, 2, 3}, # (buy,high vol), (sell,high vol), (buy,low vol), (sell,low vol)
    
    def seed(self, seed:int) -> None:
        self.np_random = Generator(PCG64DXSM(seed=seed)) #For ts creation

    def step(
        self,
        action
    ):
        T = self.T
        M = self.M
        N = self.N
        S = self.ts[:][self.time]   

        if self.time < M-1:
            self.tradingtimes[self.time:] = np.zeros(M-self.time)
            for i in range(self.time+1,M):
                self.tradingtimes[i] = i-self.time // action
            if self.time in self.tradingtimes:
                if obs in [0,2]:
                    self.cost += 1
                    self.pos += 1/self.ts[i]
                else:
                    self.cost -= 1
                    self.pos -= 1/self.ts[i]

            self.time += 1

            Snext = self.tsSS[self.time]
            
            xi = (Snext[0]**2+Snext[1]**2)/(Snext[0]+Snext[1])-(S[0]**2+S[1]**2)/(S[0]+S[1])

            self.Pi.append(action)

        if self.time == M-1:
            self.terminated = True
            self.truncated = True
                   
        self.info = {'terminal_observation': [self.reward, self.time, self.terminated, self.truncated]}
        obs = np.array([[S[i]] for i in range(len(S))])

        return obs, self.reward, self.terminated, self.truncated, self.info
        
    def reset(
        self,
        *,
        seed: Optional[int] = None
    ):
        T = self.T
        dT = self.dT
        N = self.N
        M = self.M
        self.time = self.start_time # the current time is zero
        self.p = np.zeros(N)
        self.reward = 0
        self.p = np.zeros(4*N) # current position in each option
        self.Pi = []
        mu = self.mu
        sigma = self.sigma
        
        if self.Dynamics == 'BS':
            eps0 = [dT*(mu[0]+self.np_random.normal(0,sigma[0]**2)) for i in range(M)]
            eps1 = [dT*(mu[1]+self.np_random.normal(0,sigma[1]**2)) for i in range(M)]
        
        self.ts = [[self.S0[0]*np.exp(sum(eps0[:i])), self.S0[1]*np.exp(sum(eps1[:i]))] for i in range(M)]

        S = self.ts[:][0]
        obs = np.array([[S[i]] for i in range(len(S))])

        self.terminated = False
        self.truncated = False

        return obs, {}

    def render(self):
        pass