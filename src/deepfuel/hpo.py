# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 08:46:18 2025

@author: Rodolfo Freitas
"""

import os
import json
import joblib
import numpy as np
import optuna
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
import torch
from sklearn.base import clone
import warnings
warnings.filterwarnings(
    "ignore",
    message="Choices for a categorical distribution should be a tuple of None, bool, int, float and str"
)   

# --------------------------------------------------------------------------
# Utility scoring function
# --------------------------------------------------------------------------
def compute_metric(y_true, y_pred, metric="r2"):
    if metric == "r2":
        return r2_score(y_true, y_pred)
    elif metric == "mae":
        return -mean_absolute_error(y_true, y_pred)
    elif metric == "rmse":
        return -np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric == "mse":
        return -mean_squared_error(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric '{metric}'")


def make_json_safe(obj):
    """
    Recursively convert an object to a JSON-serializable form.
    - dicts → recursively applied
    - lists/tuples → recursively applied
    - non-serializable objects → converted to str
    - primitives (int, float, str, bool, None) → unchanged
    """
    # Primitive types are safe
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    
    # Dictionaries → recursively safe
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    
    # Lists or tuples → recursively safe
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    
    # Sets → convert to list
    elif isinstance(obj, set):
        return [make_json_safe(v) for v in obj]
    
    # All other objects → convert to string
    else:
        return str(obj)


# --------------------------------------------------------------------------
# Main HPO function (sklearn-compatible for ALL models)
# --------------------------------------------------------------------------
def tune(
    model,
    suggest_params,
    X,
    y,
    metric="r2",
    n_splits=5,
    n_trials=50,
    save_dir=None,
    study_name=None,
    storage=None,
    random_state=0,
    use_cuda=True,
    verbose=True,
):
    """
    Hyperparameter optimization for ANY sklearn-compatible model.

    Parameters
    ----------
    model : sklearn-style estimator
        Must implement fit(), predict(), and set_params().
    suggest_params : callable
        Function(trial) -> dict of hyperparameters.
    X : array-like or list (including SMILES for GNNRegressor)
    y : array-like
    metric : str
        One of: 'r2', 'mae', 'rmse', 'mse'
    n_splits : int
        K-Fold splits
    n_trials : int
        Optuna trials
    save_dir : str or None
        If set, saves:
            best_params.json
            best_model.joblib (sklearn models)
            best_model.pt     (PyTorch models)
            study.pkl
    study_name : str or None
        For Optuna storage
    storage : str or None
        Optuna DB backend, e.g. "sqlite:///deepfuel_hpo.db"
    random_state : int
    use_cuda : bool, default=True
        Use GPU if available.
    verbose : bool

    Returns
    -------
    study : optuna.Study
    best_model : fitted model with best parameters
    """

    # ---------------------------------------------------
    # Create Optuna study
    # ---------------------------------------------------
    n_gpus = torch.cuda.device_count() if use_cuda else 0
    device = torch.device("cuda:0" if n_gpus > 0 else "cpu")
    
    MAX_METRICS = {"r2"}
    
    pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=1,
    interval_steps=1,
    )

    
    study = optuna.create_study(
        direction="maximize" if metric in MAX_METRICS else "minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
        pruner=pruner,)
    
    def subset(X, idx):
        if isinstance(X, np.ndarray):
            return X[idx]
        return [X[i] for i in idx]
    
    # ---------------------------------------------------
    # Objective function
    # ---------------------------------------------------
    def objective(trial):
        params = suggest_params(trial)
        
        if n_gpus > 1:
            gpu_id = trial.number % n_gpus
            trial_device = torch.device(f"cuda:{gpu_id}")
        else:
            trial_device = device

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(y)))):
            # Create a fresh model clone
            model_tr = clone(model)
            model_tr.set_params(**params)
            
            if hasattr(model_tr, "to"):  # PyTorch models
                model_tr.to(trial_device)
            
            X_train = subset(X, train_idx)
            X_val   = subset(X, val_idx)
            y_train = subset(y, train_idx)
            y_val = subset(y, val_idx)

            model_tr.fit(X_train, y_train)
            preds = model_tr.predict(X_val)

            score = compute_metric(y_val, preds, metric)
            scores.append(score)
            
            # Pruning hook
            trial.report(np.mean(scores), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    # ---------------------------------------------------
    # Run optimization
    # ---------------------------------------------------
    if n_gpus <= 1:
        # Single GPU or CPU
        n_jobs = 1
    else:
        # Multi-GPU: one trial per GPU
        n_jobs = min(n_gpus, n_trials)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose, n_jobs=n_jobs)

    best_params = study.best_params
    best_model = model.__class__(**best_params)
    best_model.fit(X, y)

    # ---------------------------------------------------
    # Saving results
    # ---------------------------------------------------
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        # Save best params
        safe_best_params = make_json_safe(best_params)

        # Save best params
        with open(os.path.join(save_dir, "best_params.json"), "w") as f:
            json.dump(safe_best_params, f, indent=4)

        # Save Optuna study
        joblib.dump(study, os.path.join(save_dir, "study.pkl"))

        # Save model
        try:
            joblib.dump(best_model, os.path.join(save_dir, "best_model.joblib"))
        except Exception:
            # PyTorch or other models
            if hasattr(best_model, "save"):
                best_model.save(os.path.join(save_dir, "best_model.pt"))
            else:
                joblib.dump(best_model, os.path.join(save_dir, "best_model.pkl"))

        if verbose:
            print(f"\n Best params saved to {save_dir}/best_params.json")
            print(f" Best model saved to {save_dir}")

    return study, best_model



