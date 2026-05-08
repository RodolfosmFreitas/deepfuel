# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 10:55:40 2025

@author: Rodolfo Freitas
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.optimize import Bounds, minimize


def optimize_fuel(obj_fun,
                  args=(),
                  optimizer='SLSQP',
                  n_features=None,
                  x0_sampler=None, 
                  bounds=None,
                  constraints=None,
                  jac=None, 
                  n_starts=20,
                  n_jobs=-1,
                  maxiter=1000,
                  random_seed=None):
    """

    Args:
        obj_fun     : objective function obj_fun(x, phi, target, alpha, beta, model)
        args         : tuple of extra arguments to pass to obj_fun
        n_features   : dimension of x (required if x0_sampler is None)
        x0_sampler  : callable that returns random x0
        constraints : list of scipy constraints or dict (user-defined)
        bounds      : scipy Bounds object or list of (min, max) tuples
        n_starts    : number of random initializations
        n_jobs      : number of parallel threads
        jac         : analytic gradient (optional) None = 2-point finite difference estimation with an absolute step size
        maxiter     : maximum iterations
        
        random_seed : int, for reproducibility

    Returns:
        best_x      : best composition found
        best_loss   : objective value
        all_solutions : list of (x_opt, loss) pairs
    """

    if random_seed is not None:
        np.random.seed(random_seed)
        
    if x0_sampler is None:
        if n_features is None:
            raise ValueError("n_features must be provided if x0_sampler is None")
        def x0_sampler(n):
            return np.random.dirichlet(np.ones(n))
    
    if bounds is None:
        bounds = Bounds(np.zeros(n_features), np.ones(n_features))
    
    if constraints is None:
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]

    # Worker executed in parallel
    def worker(seed):
        if seed is not None:
            np.random.seed(seed)
        x0 = x0_sampler(n_features)

        sol = minimize(
            fun=obj_fun,
            x0=x0,
            args=args,
            method=optimizer,
            jac=jac,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-9}
        )
        return sol.x, sol.fun

    futures = []
    solutions = []

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for i in range(n_starts):
            seed_i = random_seed + i if random_seed is not None else None
            futures.append(executor.submit(worker, seed_i))

        for f in as_completed(futures):
            solutions.append(f.result())

    # Select best solution
    best_x, best_loss = min(solutions, key=lambda t: t[1])

    return best_x, best_loss, solutions

