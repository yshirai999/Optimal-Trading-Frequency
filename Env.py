import numpy as np
from numpy.random import PCG64DXSM, Generator
import gymnasium as gym
from gymnasium.spaces import Discrete
from typing import Optional

class LocalVol(gym.Env):

    def __init__(self, Dynamics: str = 'BS', T: float = 1, dT: float = 1/252, r: float = 0, mu: list[float] = [0.08,-0.02], sigma: list[float] = [0.05, 0.5], 
                 P : list[list[float]] = [[0.8,0.05,0.1,0.05],[0.7,0.05,0.15,0.1],[0.5,0.2,0.2,0.1],[0.3,0.2,0.2,0.3]]):
        self.start_time = 0
        self.T = T # time horizon (in years)
        self.dT = dT # time step (in years e.g. 1/252 for daily)
        self.M = int(T/dT) # number of time steps in the episode (e.g. 252 for daily)
        self.tradingtimes = np.ones(self.M) # trading times (in time steps)
        self.N = len(mu) # number of scenarios
        self.r = r # risk-free rate (annualized)
        self.P = P # transition matrix for the Markov chain of drifts and volatilities
        self.MP = [] # Markov chain of observed drifts and volatilities (in the form of indices in mu and sigma)
        
        self.Dynamics = Dynamics # Black Scholes with regime switching by default (other dynamics need to be implemented)
        self.S0 = 100 # initial stock price (in investment units)
        self.ts = [] # time series of stock prices (in investment units)
        self.time = 0 # current time step
        self.terminated = False # whether the episode is terminated (i.e. when the time horizon is reached)
        self.truncated = False # whether the episode is truncated (not really implemented here, unless we add a maximum to the amount of losses the agent can take)
        self.reward = 0 # current reward
        self.info = {} # additional information
        self.sigma = sigma # volatility
        self.mu = mu # drift

        self.cash = 0 # cash position (in investment units)
        self.pos = 0 # position in the stock (in investment units)

        self.action_space = Discrete(self.M, start=dT, seed = 42) # action space is the trading frequency (in time steps)
        # The observation space is the discrete set {0,1,...,N**2}, where N is number of scenarios.
        # Thus, for instance, 0 corresponds to buy if mu[0]>0 and sell if mu[0]<0, and high vol if sigma[0]>sigma[1] and low vol if sigma[0]<sigma[1]
        self.observation_space = Discrete(self.N**2, start=0, seed=42) 
    
    def seed(self, seed:int) -> None:
        self.np_random = Generator(PCG64DXSM(seed=seed)) #For ts creation

    def step(
        self,
        action
    ):
        M = self.M
        S = self.ts[self.time]   
        obs = self.MP[self.time][0]*self.MP[self.time][1] + self.MP[self.time][1]

        if self.time < M-1:
            self.tradingtimes[self.time:] = np.zeros(M-self.time)
            for i in range(self.time+1,M):
                self.tradingtimes[i] = i-self.time // action
            if self.time in self.tradingtimes:
                if self.mu[self.MP[self.time][0]] > 0: # buy if drift is positive
                    self.cash -= 1 # cash position decreases by 1 investment unit (e.g. by 1 dollar)
                    self.pos += 1/S # position in the stock grows by 1 investment unit (e.g. by 1 dollar)
                else: # sell if drift is negative
                    self.cash += 1
                    self.pos -= 1/S

            self.time += 1
            dS = self.ts[self.time+1] - S # stock price variation
            self.cash = self.cash(1+self.r*self.dT) # cash position is updated with the risk-free rate
            self.pos = self.pos*(1+dS/S) # position in the stock is updated with the stock price variation
            self.reward = self.cash + self.pos # reward is the sum of the cash position and the stock position
        else:
            self.reward = 0
            self.terminated = True
            self.truncated = True
                   
        self.info = {'terminal_observation': [self.reward, self.time, self.terminated, self.truncated]}

        return obs, self.reward, self.terminated, self.truncated, self.info
        
    def reset(
        self,
        *,
        seed: Optional[int] = None
    ):
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
            self.MP[0] = [0,0] # initial drift and volatility are mu[0] and sigma[0]
            for i in range(M):
                self.MP[i][0] = self.np_random.random(range(self.N))
                self.MP[i][1] = self.np_random.random(range(self.N))
            eps = [dT*(mu[self.MP[i][0]]+self.np_random.normal(0,sigma[self.MP[i][1]]**2)) for i in range(M)]
        else:
            print('Dynamics not implemented')
            return
        self.ts = [self.S0*np.exp(sum(eps[:i])) for i in range(M)]

        obs = self.MP[0]

        self.terminated = False
        self.truncated = False

        return obs, {}

    def render(self):
        pass