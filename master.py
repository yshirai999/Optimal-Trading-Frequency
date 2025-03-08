##########################################
### To run this file: & PYTHONPATH PATH
### PYTHONPATH is the path to python.exe
### in the OTF conda environment
### PATH is this python file's path
### Example: & C:/Users/yoshi/Anaconda3/envs/OTF/python.exe C:/Users/yoshi/OneDrive/Desktop/Research/Optimal-Trading-Frequency/master.py
##########################################
### Libraries
##########################################

from env import LocalVol
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from loggers import TensorboardCallback
import numpy as np
import random
import os
import matplotlib.pyplot as plt

##########################################
### Train/load model
##########################################

N = 10
Dynamics  = 'BS'
star_time = 0
T = 1
dT = 1/252
r = 0
mu = [0.05]
sigma = [0.3]
P = [[1]]

LVol = LocalVol(Dynamics = Dynamics, T = T, dT = dT, mu = mu, sigma = sigma, P = P)
LVol.seed(seed=random.seed(10))

env = gym.wrappers.TimeLimit(LVol, max_episode_steps=T)
env = Monitor(env, allow_early_resets=True)

steps = 10000

path_folder = f"C:/Users/yoshi/OneDrive/Desktop/Research/Optimal-Trading-Frequency/BS_PPO" # PATH to the BS_PPO_Models folder
path = f"{path_folder}/BS_PPO_{str(steps)}_n_regimes_{str(len(mu))}"
for k in range(len(mu)):
    path += f"mu[{str(k)}]_mu{str(int(mu[k]*100))}"

eval_callback = EvalCallback(env, best_model_save_path=path_folder,
                             log_path=path_folder, eval_freq=500,
                             deterministic=True, render=False)

try:
    model = PPO.load(path, env = DummyVecEnv([lambda: env]), print_system_info=True)
except:
    print("Training model...")
    if not os.path.exists(f"{path_folder}/tensorboard/"):
        os.makedirs(f"{path_folder}/tensorboard/")
    model = PPO('MlpPolicy', DummyVecEnv([lambda: env]), learning_rate=0.001, verbose=1,
                tensorboard_log=f"{path_folder}/tensorboard/")
    model.learn(total_timesteps=steps, callback=eval_callback, log_interval = 100)
    model.save(f"{path}.zip")
    # To open tensorboard window, run tensorboard --logdir C:/Users/yoshi/OneDrive/Desktop/Research/Optimal-Trading-Frequency/BS_PPO/tensorboard/PPO_1

##########################################
### Experiment
##########################################

Nepisodes = 100
rew = []
act = []
tradingtimes = []

for i in range(Nepisodes):
    obs = env.reset()
    # obs = [[obs[0][i] for i in range(len(obs[0]))]]
    cont = True
    i = 0
    act.append([])
    tradingtimes.append([])
    reward_episode = 0
    while cont:
        action = 1#env.action_space.sample()
        obs, reward, terminated, truncated, info = LVol.step(action)
        act[-1].append(action)
        reward_episode += reward
        i += 1
        if any([terminated, truncated]):
            cont = False
            tradingtimes[-1].append(env.unwrapped.tradingtimes)
            rew.append(reward_episode)

print(np.mean(rew),np.std(rew))