"""Compute common regression metrics."""

import numpy as np
from mapie.metrics.regression import (
    regression_coverage_score,
    regression_mean_width_score,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)).item(),
        "mae": mean_absolute_error(y_true, y_pred)}

def conformal_metrics(y_test, predicted_intervals):
    
        return {
            "coverage_score": regression_coverage_score(y_test, predicted_intervals),
            "mean_width_score": regression_mean_width_score(predicted_intervals)}