# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:13:29 2026

@author: Rodolfo Freitas
"""
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV

#%% Define the problem
class FuelBlendProblem(ElementwiseProblem):
    def __init__(self, obj_fun, n_comp, n_obj, bounds_oper=None, args=(), constraints_fun=None):
        """
        n_comp: number of composition variables
        bounds_oper: tuple of (lower_oper, upper_oper) for operational variables (T0, phi, P)
        """
        self.n_comp = n_comp
        self.bounds_oper = bounds_oper
        self.n_oper = 0 if bounds_oper is None else bounds_oper[0].shape[0]
        self.bounds_oper = bounds_oper
        self.obj_fun = obj_fun
        self.constraints_fun = constraints_fun
        self.args = args
        
        n_var = self.n_comp + self.n_oper
        
      
        # --- Bounds ---
        if self.n_oper > 0:
            xl = np.concatenate([np.zeros(n_comp), bounds_oper[0]])
            xu = np.concatenate([np.ones(n_comp), bounds_oper[1]])
        else:
            xl = np.zeros(n_comp)
            xu = np.ones(n_comp)
        
       # --- Constraints ---
        if constraints_fun is not None:
            if self.n_oper > 0:
                x0 = np.concatenate([np.ones(n_comp)/n_comp, bounds_oper[0]])
            else:
                x0 = np.ones(n_comp) / n_comp

            n_constr = len(np.atleast_1d(constraints_fun(x0)))
        else:
            n_constr = 0
        
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl,
            xu=xu
        )
        
        
        

    def _evaluate(self, x, out, *args, **kwargs):
        # --- Composition ---
        x_comp = x[:self.n_comp]
        x_comp = np.maximum(x_comp, 1e-12)
        x_comp /= np.sum(x_comp)

        # --- Operational (if exists) ---
        if self.n_oper > 0:
            x_oper = x[self.n_comp:]
            x_full = np.concatenate([x_comp, x_oper])
        else:
            x_full = x_comp
        
        # Objective
        f = self.obj_fun(x_full, *self.args)
        out["F"] = np.atleast_1d(f)
        
        # Constraints
        if self.constraints_fun is not None:
            g = self.constraints_fun(x_full)
            out["G"] = np.atleast_1d(g)

#%% Sampling and Repair
class DirichletSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        # --- Composition ---
        comp = np.random.dirichlet(np.ones(problem.n_comp), size=n_samples)

        # --- Operational (if exists) ---
        if problem.n_oper > 0:
            lower_oper, upper_oper = problem.bounds_oper
            oper = np.random.uniform(lower_oper, upper_oper, size=(n_samples, problem.n_oper))
            return np.hstack([comp, oper])
        else:
            return comp

class SimplexRepair(Repair):
    def _do(self, problem, X, **kwargs):
        # --- Composition repair ---
        X[:, :problem.n_comp] = np.maximum(X[:, :problem.n_comp], 1e-12)
        X[:, :problem.n_comp] /= np.sum(X[:, :problem.n_comp], axis=1, keepdims=True)

        # --- Operational repair (if exists) ---
        if problem.n_oper > 0:
            lower_oper, upper_oper = problem.bounds_oper
            X[:, problem.n_comp:] = np.clip(X[:, problem.n_comp:], lower_oper, upper_oper)

        return X

#%% Optimization wrapper

class FuelOptimizer:
    """
    Multi-objective fuel optimization using GA (NSGA-II / NSGA-III).

    Supports:
        - composition-only optimization
        - composition + operational variables
        - custom sampling (e.g., RL-guided)

    Parameters
    ----------
    obj_fun : callable
    n_comp : int
    n_obj : int
    bounds_oper : tuple or None
    constraints_fun : callable, optional
    pop_size : int
    n_gen : int
    sampling : pymoo Sampling, optional
    seed : int, optional
    """

    def __init__(self,
                 obj_fun,
                 n_comp,
                 n_obj,
                 bounds_oper=None,
                 args=(),
                 constraints_fun=None,
                 pop_size=100,
                 n_gen=200,
                 sampling=None,
                 seed=None,
                 hv_ref_point=None,     
                 callback=None): 

        self.obj_fun = obj_fun
        self.n_comp = n_comp
        self.n_obj = n_obj
        self.bounds_oper = bounds_oper
        self.args = args
        self.constraints_fun = constraints_fun
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.sampling = sampling
        self.seed = seed
        self.hv_ref_point = hv_ref_point
        self.callback = callback

        self.problem = None
        self.algorithm = None
        self.result = None
        self.hv_indicator = None   
        self.hv_history = []
        self.F_history = []
        self.X_history = []

    def _build_problem(self):
        self.problem = FuelBlendProblem(
            obj_fun=self.obj_fun,
            n_comp=self.n_comp,
            n_obj=self.n_obj,
            bounds_oper=self.bounds_oper,
            args=self.args,
            constraints_fun=self.constraints_fun
        )
        

    def _build_algorithm(self):
        # Sampling
        sampling = self.sampling if self.sampling is not None else DirichletSampling()

        # Operators
        crossover = SBX(prob=0.9, eta=15)
        mutation = PM(eta=20)
        repair = SimplexRepair()
        
        if self.hv_ref_point is not None:
            self.hv_indicator = HV(ref_point=self.hv_ref_point)

        # Algorithm selection
        if self.n_obj <= 2:
            self.algorithm = NSGA2(
                pop_size=self.pop_size,
                sampling=sampling,
                crossover=crossover,
                mutation=mutation,
                repair=repair,
                eliminate_duplicates=True
            )
            algo_name = "NSGA-II"
        else:
            ref_dirs = get_reference_directions("das-dennis", self.n_obj, n_partitions=12)
            self.algorithm = NSGA3(
                pop_size=self.pop_size,
                ref_dirs=ref_dirs,
                sampling=sampling,
                crossover=crossover,
                mutation=mutation,
                repair=repair,
                eliminate_duplicates=True
            )
            algo_name = "NSGA-III"
              

        mode = "composition-only" if self.problem.n_oper == 0 else "composition + operational"
        print(f"Using {algo_name} ({mode}), n_obj={self.n_obj}")

    def run(self):
        """
        Execute the optimization.
        
        Returns
        -------
        res : pymoo Result object
        """
        # Build components
        self._build_problem()
        self._build_algorithm()
        
        def _callback(algorithm): 
            pop = algorithm.pop
            cv = pop.get("CV")
            rank = pop.get("rank")

            F = pop.get("F")
            X = pop.get("X")

            # safety reshape (prevents silent bugs)
            cv = np.atleast_1d(cv).flatten()
            rank = np.atleast_1d(rank).flatten()

            F = np.atleast_2d(F)
            X = np.atleast_2d(X)

            # check alignment
            n = F.shape[0]

            if len(cv) != n or len(rank) != n:
                raise ValueError(
                    f"Shape mismatch: F={F.shape}, CV={cv.shape}, rank={rank.shape}"
                )

            feasible = cv <= 0
            nd_mask = (rank == 0) & feasible
            
            F_nd = F[nd_mask]
            X_nd = X[nd_mask]
            
            # store ONLY Pareto front (clean + efficient)
            self.F_history.append(F_nd.copy())
            self.X_history.append(X_nd.copy())
            
            # hypervolume
            if self.hv_indicator is not None and len(F_nd) > 0:
                hv = self.hv_indicator(F_nd)
                self.hv_history.append(hv)
            else:
                self.hv_history.append(np.nan)
                    
            # external callback
            if self.callback is not None: 
                self.callback(algorithm)
                        

        # Run optimization
        self.result = minimize(
            self.problem,
            self.algorithm,
            termination=('n_gen', self.n_gen),
            seed=self.seed,
            verbose=True,
            callback=_callback
        )

        return {
            "result": self.result,
            "hv_histroy": self.hv_history,
            "pareto": self.F_history,
            "designs": self.X_history}
