# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:07:17 2026

@author: Rodolfo Freitas
"""


import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import DDPG, PPO, SAC, TD3

device = "cuda" if torch.cuda.is_available() else "cpu"

class FuelRL_MOEnv(gym.Env):
    
    """
    Multi-objective Reinforcement Learning environment for DeepFuel platform.

    State:
        - n_comp fuel fractions + n_oper operational variables

    Action:
        - Same as state: first n_comp entries are fuel fractions (projected to simplex)
        - Remaining n_oper entries are operational variables (clipped to bounds)

    Objectives:
        - obj_fun(x) returns vector of objectives
        - Always minimized; objectives to maximize should be negated
        - Scalarization function converts multi-objective vector into scalar reward

    Constraints:
        - constraints_fun(x) returns g_i(x) values
        - Reward penalizes constraint violations: g_i(x) > 0

    Step function:
        - step(action) returns (next_state, reward, done, info)
        - One-step environment by default
        - next_state = projected composition + clipped operational variables
        - reward = scalarization(F) - penalty * sum(G[G>0])
        - done = True (one-step environment)
        - info contains 'F', 'G', and 'x'
        

    Methods:
        train(method="PPO", timesteps=5000):
            Train an RL agent (PPO, SAC, DDPG, TD3) directly on this environment.
    """
    
    metadata = {'render.modes': ['human']}
    def __init__(self, n_comp, bounds_oper=None, obj_fun=None, constraints_fun=None, scalarization=None, penalty_factor=1):
        super().__init__()
        self.n_comp = n_comp
        self.bounds_oper = bounds_oper
        self.n_oper = 0 if bounds_oper is None else bounds_oper[0].shape[0]
        self.obj_fun = obj_fun
        self.constraints_fun = constraints_fun
        self.scalarization = scalarization if scalarization else lambda F: -np.sum(F)
        self.penalty_factor = penalty_factor
        
        # Define action and observation space
        if self.n_oper > 0:
            low = np.concatenate([np.zeros(self.n_comp), bounds_oper[0]])
            high = np.concatenate([np.ones(self.n_comp), bounds_oper[1]])
        else:
            low = np.zeros(self.n_comp)
            high = np.ones(self.n_comp)
        
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = self.action_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize composition on simplex
        comp = np.random.dirichlet(np.ones(self.n_comp))
        # operational
        if self.n_oper > 0:
            oper = np.random.uniform(self.bounds_oper[0], self.bounds_oper[1])
            self.state = np.concatenate([comp, oper]).astype(np.float32)
        else:
            self.state = comp.astype(np.float32)
    
        return self.state, {}  #  must return (obs, info)

    def step(self, action):
        
        # Project fuel fractions to simplex
        x_comp = np.maximum(action[:self.n_comp], 1e-12)
        x_comp /= np.sum(x_comp)
        
        # Clip operational variables if they exist
        if self.n_oper > 0:
            x_oper = np.clip(action[self.n_comp:], self.bounds_oper[0], self.bounds_oper[1])
            x_full = np.concatenate([x_comp, x_oper]).astype(np.float32)
        else:
            x_full = x_comp.astype(np.float32)
       
        # Compute objectives and constraints
        F = np.array(self.obj_fun(x_full))
        G = np.array(self.constraints_fun(x_full)) if self.constraints_fun else np.zeros_like(F)
    
        # Reward: scalarized objectives minus penalty for constraint violations
        reward = self.scalarization(F) - self.penalty_factor * np.sum(G[G > 0])
    
        terminated = True      # one-step episode
        truncated = False      # no time limit
    
        self.state = x_full
        info = {'F': F, 'G': G, 'X': x_full}
    
        return x_full, reward, terminated, truncated, info

    def render(self, mode='human'):
        print("Current state:", self.state)
    
    def train(self, method="PPO", timesteps=5000, verbose=1):
        """
        Train an RL agent inside this environment.
        method : str, one of "PPO", "SAC", "DDPG", "TD3"
        timesteps : int, number of training steps
        """
        if method == "PPO":
            model = PPO("MlpPolicy", self, verbose=verbose, device=device)
        elif method == "SAC":
            model = SAC("MlpPolicy", self, verbose=verbose, device=device)
        elif method == "DDPG":
            model = DDPG("MlpPolicy", self, verbose=verbose, device=device)
        elif method == "TD3":
            model = TD3("MlpPolicy", self, verbose=verbose, device=device)
        else:
            raise ValueError("method must be 'PPO', 'SAC', 'DDPG', or 'TD3'")
        
        model.learn(total_timesteps=timesteps)
        return model


