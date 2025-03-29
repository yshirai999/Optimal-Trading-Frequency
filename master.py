##########################################
### To run this file: & PYTHONPATH PATH
### PYTHONPATH is the path to python.exe
### in the OTF conda environment
### PATH is this python file's path
##########################################
### Libraries
##########################################

from LocalVolEnv import LocalVol
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from Loggers import TensorboardCallback
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import torch

##########################################
### Train/load model
##########################################

N = 10
Dynamics  = 'BS'
star_time = 0
T = 1
dT = 1/252
r = 0
mu = [0.05,-0.02]
sigma = [0.3,0.3]
P = [[0.9,0.0,0.0,0.1],[0.9,0.0,0.0,0.1],[0.5,0.0,0.0,0.5],[0.5,0.0,0.0,0.5]]
cuda = True #Use cuda device and larger network architecture (3 layers, 256 neurons per layer) and larger batch size

LVol = LocalVol(Dynamics = Dynamics, T = T, dT = dT, mu = mu, sigma = sigma, P = P)
LVol.seed(seed=random.seed(10))

env = gym.wrappers.TimeLimit(LVol, max_episode_steps=10000)
env = Monitor(env, allow_early_resets=True)

steps = 1000000

base_path = os.path.dirname(os.getcwd()) 
path_folder = os.path.join(base_path,'Optimal-Trading-Frequency','BS_PPO') # PATH to the BS_PPO_Models folder
path = f"{path_folder}/BS_PPO_{str(steps)}_n_regimes_{str(len(mu))}"

for k in range(len(mu)):
    path += f"_mu[{str(k)}]={str(int(mu[k]*100))}_P[{str(k)}]={str(int(P[0][k]*100))}_sigma[{str(k)}]={str(int(sigma[k]*100))}" # Add mu and sigma to the path for each regime
    # This will help to identify the model trained for each set of parameters
    # Example: path = .../BS_PPO_1000000_n_regimes_2_mu[0]=5_P[0]=90_sigma[0]=10_mu[1]=-2_P[1]=0_sigma[1]=10

if cuda:
    path += 'cuda'

eval_callback = EvalCallback(env, best_model_save_path=path_folder,
                             log_path=path_folder, eval_freq=500,
                             deterministic=True, render=False)

try:
    model = PPO.load(path, env = DummyVecEnv([lambda: env]), print_system_info=True)
except:
    print("Training model...")
    if not os.path.exists(f"{path_folder}/tensorboard/"):
        os.makedirs(f"{path_folder}/tensorboard/")
    if cuda:
        
        print(torch.cuda.is_available())  # Should return True
        print(torch.cuda.device_count())  # Number of GPUs available
        print(torch.cuda.get_device_name(0))  # Name of the GPU

        policy_kwargs = dict(net_arch=[256, 256, 256])
        model = PPO('MlpPolicy', DummyVecEnv([lambda: env]), learning_rate=0.001, verbose=1, batch_size=1024, 
                    policy_kwargs=policy_kwargs, device="cuda",tensorboard_log=f"{path_folder}/tensorboard/")
    else:
        model = PPO('MlpPolicy', DummyVecEnv([lambda: env]), learning_rate=0.001, verbose=1,
                    tensorboard_log=f"{path_folder}/tensorboard/")
    model.learn(total_timesteps=steps, callback=eval_callback, log_interval = 100)
    model.save(f"{path}.zip")

##########################################
### Experiment
##########################################

Nepisodes = 100
rew = []
act = []
tradingtimes = []

for i in range(Nepisodes):
    obs = env.reset()
    obs = obs[0]
    cont = True
    i = 0
    act.append([])
    tradingtimes.append([])
    reward_episode = 0
    while cont:
        action = model.predict([obs], deterministic = True)
        obs, reward, terminated, truncated, info = LVol.step(action[0][0])
        act[-1].append(action)
        reward_episode += reward
        i += 1
        if any([terminated, truncated]):
            cont = False
            tradingtimes[-1].append(env.unwrapped.tradingtimes)
            rew.append(reward_episode)

print(np.mean(rew),np.std(rew))

# obs = Benv.reset()
#     obs = [[obs[0][i] for i in range(len(obs[0]))]]
#     cont = True
#     i = 0
#     while cont:
#         action, _states = model.predict(obs, deterministic = True)