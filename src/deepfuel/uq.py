# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 16:47:44 2025

@author: Rodolfo Freitas
"""

import numpy as np
from sklearn.model_selection import train_test_split
from mapie.regression import (
    SplitConformalRegressor,
    CrossConformalRegressor,
    JackknifeAfterBootstrapRegressor,
    ConformalizedQuantileRegressor,
)
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol


class ConformalizedBayesRegressor:
    """
    Conformalized predictions for probabilistic models (e.g., Gaussian Process, Bayesian NNs).

    This method calibrates uncertainty estimates using conformal prediction
    based on normalized residuals (|y - μ| / σ), where μ and σ come from a
    probabilistic model’s predictive mean and standard deviation.

    Parameters
    ----------
    model : object
        Probabilistic model with `fit(X, y)` and `predict(X, return_std=True)` methods.
    confidence_level : float, default=0.95
        Desired confidence level for the prediction intervals.
    calibration_size : float, default=0.2
        Proportion of data to use for calibration (the rest for training).
    random_state : int or None
        Random seed for reproducible calibration split.
    """

    def __init__(self, model, confidence_level=0.95, calibration_size=0.2, random_state=None):
        if not hasattr(model, "predict"):
            raise ValueError("Model must implement a `predict(X, return_std=True)` method.")
        self.model = model
        self.confidence_level = confidence_level
        self.calibration_size = calibration_size
        self.random_state = random_state

    def fit(self, X, y):
        
        self.model.fit(X, y)
        
        return self 
    
    def conformalize(self, X, y):
        # get conformal scores
        mu, sigma = self.model.predict(X, return_std=True)
        scores = np.abs((y - mu) / sigma)
        
        # get adjusted quantile
        n = len(X)
        q_level = np.ceil((n+1)*(self.confidence_level)) / n
        self.q_hat = np.quantile(scores, q_level, method='higher')
        return self
    

    def predict_interval(self, X, alpha=None):
        """
        Predicts the mean and conformal prediction intervals.

        Returns
        -------
        mu : np.ndarray
            Predictive mean.
        intervals : np.ndarray of shape (n_samples, 2)
            Lower and upper prediction interval bounds.
        """
        if not hasattr(self, "q_hat"):
            raise RuntimeError("Model must be fitted before prediction (call `.fit()`).")

        if alpha is None:
            alpha = 1 - self.confidence_level

        mu, sigma = self.model.predict(X, return_std=True)
       

        lower = mu - self.q_hat * sigma
        upper = mu + self.q_hat * sigma

        return mu, np.vstack([lower, upper]).T


def conformalized_model(model=None, conformal='split', **kwargs):
    """
    Factory function to return conformalized regressors (MAPIE or Bayesian).

    Parameters
    ----------
    model : estimator
        Base model (must implement `fit` and `predict`).
    conformal : str, default='split'
        Conformalization strategy: 'split', 'cross', 'jackknife', 'cqr', or 'bayes'.
    kwargs : dict
        Additional parameters for the chosen conformalizer.

    Returns
    -------
    Conformalized regressor instance.
    """
    conformalizers = {
        "split": SplitConformalRegressor,
        "cross": CrossConformalRegressor,
        "jackknife": JackknifeAfterBootstrapRegressor,
        "cqr": ConformalizedQuantileRegressor,
        "bayes": ConformalizedBayesRegressor,
    }

    if conformal not in conformalizers:
        raise ValueError(f"Unknown conformalizer '{conformal}'. Available: {list(conformalizers.keys())}")

    conformalizer_cls = conformalizers[conformal]

    if conformal == "bayes":
        confidence_level = kwargs.pop("confidence_level", 0.95)
        calibration_size = kwargs.pop("calibration_size", 0.2)
        random_state = kwargs.pop("random_state", None)
        return conformalizer_cls(model, confidence_level, calibration_size, random_state)

    return conformalizer_cls(estimator=model, **kwargs)



def sobol_sensitivity_analysis(model, num_features, bounds=None, N=512, feature_names=None):
    """
    Perform Sobol sensitivity analysis for a PyTorch Geometric GNN model.
    
    Parameters
    ----------
    model : model
        Trained model that takes features as input.
    num_features : int
        Number of input features for the model.
    bounds : list of [min, max], optional
        Bounds for each input feature, default [0,1] for all features.
    N : int, default=512
        Base sample size for Saltelli sampling.
    feature_names : list of str, optional
        Names of input features for reporting.
    
    Returns
    -------
    Si : dict
        Sobol sensitivity indices with keys 'S1', 'ST', 'S2', 'S1_conf', 'ST_conf', 'S2_conf'.
    """
    
    # Define problem
    if bounds is None:
        bounds = [[0, 1]] * num_features
    if feature_names is None:
        feature_names = [f'x{i}' for i in range(num_features)]
    
    problem = {
        'num_vars': num_features,
        'names': feature_names,
        'bounds': bounds
    }
    
    # Generate Saltelli samples
    param_values = sobol_sample.sample(problem, N, calc_second_order=True)
    
    # Evaluate model
    Y = model.predict(param_values)
    
    # Perform Sobol analysis
    Si = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=False)
    
    return Si


#%% Example

# from sklearn.datasets import load_diabetes
# from deepfuel.models import get_model

# diabetes = load_diabetes()
# X, y = diabetes.data, diabetes.target

# # Build and train model
# model_name = "ridge"
# model = get_model(model_name)
# print(model)

# model.fit(X, y)

# si = sobol_sensitivity_analysis(model, num_features=X.shape[1])