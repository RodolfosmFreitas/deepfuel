# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:38:17 2026

@author: Rodolfo Freitas
"""

import numpy as np
from deepfuel.rl import FuelRL_MOEnv
from deepfuel.ga import FuelOptimizer
from stable_baselines3 import PPO, SAC, DDPG, TD3
from pymoo.core.sampling import Sampling
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# convergence callback
class ConvergenceCallback(BaseCallback):
    def __init__(self, window=20, tol=1e-3, verbose=0):
        super().__init__(verbose)
        self.window = window
        self.tol = tol
        self.rewards = []

    def _on_step(self):
        infos = self.locals.get("infos", [])

        if len(infos) > 0 and "episode" in infos[0]:
            r = infos[0]["episode"]["r"]
            self.rewards.append(r)

            if len(self.rewards) >= self.window:
                recent = self.rewards[-self.window:]
                if np.std(recent) < self.tol:
                    if self.verbose:
                        print("RL converged (reward stabilized)")
                    return False  # stop training

        return True

# --- RL-guided GA sampling ---
class RLSampling(Sampling):
    """RL-guided sampling compatible with composition-only or composition+oper."""

    def __init__(self, model_fn, n_comp, bounds_oper=None):
        super().__init__()
        self.model_fn = model_fn
        self.n_comp = n_comp
        self.bounds_oper = bounds_oper
        self.n_oper = 0 if bounds_oper is None else bounds_oper[0].shape[0]

    def _do(self, problem, n_samples, **kwargs):
        X = []

        for _ in range(n_samples):

            # --- Baseline observation ---
            comp = np.random.dirichlet(np.ones(self.n_comp))

            if self.n_oper > 0:
                oper = np.random.uniform(self.bounds_oper[0], self.bounds_oper[1])
                obs = np.concatenate([comp, oper])
            else:
                obs = comp

            obs = obs.astype(np.float32)

            # --- RL action ---
            action = self.model_fn(obs)
            action = np.array(action).flatten()

            # --- Safety check ---
            if action.shape[0] < self.n_comp:
                # fallback to random if invalid
                x = comp
                if self.n_oper > 0:
                    x = np.concatenate([comp, oper])
                X.append(x)
                continue

            # --- Composition ---
            x_comp = np.maximum(action[:self.n_comp], 1e-12)
            x_comp /= np.sum(x_comp)

            # --- Operational (if exists) ---
            if self.n_oper > 0:
                x_oper = np.clip(
                    action[self.n_comp:self.n_comp + self.n_oper],
                    self.bounds_oper[0],
                    self.bounds_oper[1]
                )
                x_full = np.concatenate([x_comp, x_oper])
            else:
                x_full = x_comp

            X.append(x_full)

        return np.array(X, dtype=np.float32)

    
# --- GA wrapper to optionally reset from existing population ---  
class GAResetWrapper(gym.Wrapper):
    def __init__(self, env, initial_population=None, ga_prob=0.7):
        super().__init__(env)
        self.initial_population = initial_population
        self.ga_prob = ga_prob

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.initial_population is not None and len(self.initial_population) > 0:
            if np.random.rand() < self.ga_prob:
                idx = np.random.randint(len(self.initial_population))
                obs = np.array(self.initial_population[idx], dtype=np.float32)

        return obs, info

# --- Hybrid GA + RL optimization ---
class HybridOptimization:
    """
    Hybrid GA + RL Optimization for Multi-Objective Fuel Design.

    This class implements a hybrid optimization loop that combines:
        1. A Genetic Algorithm (GA) to explore the fuel composition and operational space.
        2. A Reinforcement Learning (RL) agent (PPO, SAC, DDPG, or TD3) to guide
           the GA via learned policy sampling.

    Workflow:
        - Cycle 1: GA randomly initializes the population and finds Pareto-optimal solutions.
        - RL is trained on GA Pareto solutions to learn promising regions in the design space.
        - Subsequent cycles: GA uses RL-guided sampling with a probability to bias population 
          towards RL-learned high-performance regions.
        - The loop continues for `n_cycles` to iteratively improve exploration and exploitation.

    Components:
        - RLSampling: RL-guided sampling class for GA, includes softmax projection of fuel fractions.
        - GAResetWrapper: Gym wrapper to initialize RL episodes from GA solutions.
        - FuelRL_MOEnv: Multi-objective RL environment for fuel design (state = fuel fractions + operational variables).

    Parameters
    ----------
    n_comp : int
        Number of fuel components (composition variables).
    bounds_oper : tuple of np.ndarray
        Operational variable bounds as (lower_bounds, upper_bounds), shape = (n_oper,).
    obj_fun : callable
        Objective function. Input: x_full (composition + operational variables),
        Output: array of objectives (all to be minimized).
    n_obj : int, default=2
        Number of objectives in the optimization problem.
    constraints_fun : callable, optional
        Constraints function. Input: x_full, Output: array of g_i(x), 
        penalized if g_i(x) > 0.
    n_cycles : int, default=3
        Number of hybrid GA ↔ RL cycles.
    ga_params : dict, optional
        Parameters for GA: {'pop_size': int, 'n_gen': int, 'seed': int}.
    rl_params : dict, optional
        Parameters for RL training: {'method': str, 'timesteps': int, 'verbose': int}.
        method must be one of 'PPO', 'SAC', 'DDPG', 'TD3'.

    Attributes
    ----------
    model : stable_baselines3.BaseAlgorithm or None
        Trained RL agent after the last cycle.
    n_comp, bounds_oper, obj_fun, constraints_fun, n_obj, n_cycles, ga_params, rl_params, tau
        Stores the corresponding initialization parameters.

    Methods
    -------
    _train_rl(pareto_solutions)
        Trains an RL agent on the given Pareto-optimal solutions. Returns
        a callable for RL-guided sampling in GA.
    run()
        Executes the hybrid GA ↔ RL optimization loop for `n_cycles`.
        Returns the last GA result (Pareto front) and trained RL model.

    Notes
    -----
    - The GA + RL framework is especially useful for high-dimensional, multi-objective
      design problems where pure GA or pure RL may converge slowly.
    - Fuel fractions are projected to enforce the simplex constraint
      (sum = 1, all fractions positive), while operational variables are clipped
      to bounds.
    - The RL agent uses deterministic predictions for GA sampling to ensure
      reproducibility of the hybrid loop.
    """
    def __init__(self, n_comp, bounds_oper, obj_fun, n_obj=2, constraints_fun=None,
                 n_cycles=3, ga_params=None, rl_params=None):
        self.n_comp = n_comp
        self.bounds_oper = bounds_oper
        self.obj_fun = obj_fun
        self.constraints_fun = constraints_fun
        self.n_obj = n_obj
        self.n_cycles = n_cycles
        self.ga_params = ga_params if ga_params else {'pop_size': 50, 'n_gen': 50, 'seed': 42}
        self.rl_params = rl_params if rl_params else {'method': 'SAC', 'timesteps': 5000, 'verbose': 1}
        self.model = None  # RL agent

    def _train_rl(self, pareto_solutions):
        env = FuelRL_MOEnv(
            n_comp=self.n_comp,
            bounds_oper=self.bounds_oper,
            obj_fun=self.obj_fun,
            constraints_fun=self.constraints_fun)
        wrapped_env = GAResetWrapper(env, initial_population=pareto_solutions)
        
        wrapped_env = Monitor(wrapped_env)


        # RL training parameters
        method = self.rl_params.get('method', 'SAC')
        timesteps = self.rl_params.get('timesteps', 5000)
        verbose = self.rl_params.get('verbose', 1)
        
        tol = self.rl_params.get('tol', 1e-3)
        window = self.rl_params.get('window', 20)
        
        callback = ConvergenceCallback(window=window, tol=tol, verbose=verbose)

        
        # Create SB3 model
        if method == "PPO":
            model = PPO("MlpPolicy", wrapped_env, verbose=verbose, device=device)
        elif method == "SAC":
            model = SAC("MlpPolicy", wrapped_env, verbose=verbose, device=device)
        elif method == "DDPG":
            model = DDPG("MlpPolicy", wrapped_env, verbose=verbose, device=device)
        elif method == "TD3":
            model = TD3("MlpPolicy", wrapped_env, verbose=verbose, device=device)
        else:
            raise ValueError("RL method must be PPO, SAC, DDPG, or TD3")
        
        # Train    
        model.learn(total_timesteps=timesteps, callback=callback)
        self.model = model
        
        # Return callable for GA sampling (deterministic)
        def model_fn(obs):
            action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
            return action.flatten()     
        
        return model_fn

    def run(self):
        """
        Run full hybrid GA + RL optimization loop.
        Returns the last GA result and trained RL model.
        """
        res = None
        self.cycles_F = []
        self.cycles_X = []
        for cycle in range(self.n_cycles):
            print(f"\n===== Cycle {cycle+1} =====")

            # GA sampling
            if self.model is None:
                sampling = None
                print("GA: random initialization")
            else:
                # pass callable instead of RL model
                def model_fn(obs):
                    return self.model.predict(obs.reshape(1, -1), deterministic=True)[0].flatten()
                sampling = RLSampling(model_fn, self.n_comp, self.bounds_oper)
                print("GA: RL-guided sampling")
            
            # Run GA
            res = FuelOptimizer(
                obj_fun=self.obj_fun,
                n_comp=self.n_comp,
                n_obj=self.n_obj,
                bounds_oper=self.bounds_oper,
                constraints_fun=self.constraints_fun,
                pop_size=self.ga_params.get('pop_size', 50),
                n_gen=self.ga_params.get('n_gen', 50),
                sampling=sampling,
                seed=self.ga_params.get('seed', 42)).run()
            # Store Pareto front (objectives)
            self.cycles_F.append(np.array(res['result'].F))
            self.cycles_X.append(np.array(res['result'].X))
            print(f"Cycle {cycle+1}: Pareto solutions found = {len(res.X)}")

            # Train RL on GA solutions
            self._train_rl(res['result'].X)


        return res, self.model


