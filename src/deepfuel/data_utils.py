
"""
Pre-processing data
"""

from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from deepfuel.models import get_model
from typing import Optional, Union
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import RFE, RFECV

from sklearn.base import BaseEstimator, TransformerMixin

class MADScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.mad_ = np.mean(np.abs(X - self.mean_), axis=0)
        # Avoid division by zero
        self.mad_[self.mad_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X)
        return (X - self.mean_) / self.mad_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X):
        X = np.asarray(X)
        return X * self.mad_ + self.mean_

def prepare_data(
    X,
    y,
    train_size: Union[float, int],
    shuffle: bool = True,
    scaler_X: Optional[str] = None,
    scaler_y: Optional[str] = None,
    random_state: Optional[int] = None
):
    """
    Split and optionally scale data.

    Parameters
    ----------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target values.
    train_size : float or int
        Size of training dataset.
    shuffle : bool, default=True
        Shuffle the data before splitting.
    scale_data : bool, default=True
        Whether to scale the dataset.
    scaler_X : str, optional
        Scaler for input features.
    scaler_y : str, optional
        Scaler for output targets.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test, scalerX, scalery
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, shuffle=shuffle, random_state=random_state
    )

    scalers = {
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "quantile": QuantileTransformer(),
        "power": PowerTransformer(),
        "l2normalizer": Normalizer(),
        "minmax": MinMaxScaler(),
        "maxabs": MaxAbsScaler(),
        "mad": MADScaler(),
        None:None,
    }

    # Default scalers
    scalerX, scalery = None, None

    if scaler_X not in scalers:
        raise ValueError(f"Unknown scaler_X '{scaler_X}'. Available: {list(scalers.keys())}")
    if scaler_y not in scalers:
        raise ValueError(f"Unknown scaler_y '{scaler_y}'. Available: {list(scalers.keys())}")

    # Scale X
    if scaler_X is not None:
        scalerX = scalers[scaler_X]
        X_train = scalerX.fit_transform(X_train)
        X_test = scalerX.transform(X_test)

    # Scale y (ensure 2D for scalers)
    if scaler_y is not None:
        scalerY = scalers[scaler_y]
        y_train = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        y_test = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test
        y_train = scalerY.fit_transform(y_train)
        y_test = scalerY.transform(y_test)
        scalery = scalerY

    # Flatten y if originally 1D
    if y.ndim == 1:
        y_train = y_train.ravel()
        y_test = y_test.ravel()

            

    return X_train, X_test, y_train, y_test, scalerX, scalery


def remove_low_variance(X: np.ndarray, threshold: float = 0.1, verbose: bool = False):
    """
    Remove features from a NumPy array that have variance below a given threshold.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    threshold : float, default=0.1
        Features with variance below this threshold are removed.
    verbose : bool, default=True
        Whether to print removed column indices.

    Returns
    -------
    X_reduced : np.ndarray
        Feature matrix with low-variance features removed.
    kept_indices : np.ndarray
        Indices of features that were kept.
    """
    selector = VarianceThreshold(threshold)
    selector.fit(X)
    X_reduced = X[:, selector.get_support(indices=True)]
    
    if verbose:
        removed_cols = set(np.arange(X.shape[1])) - set(selector.get_support(indices=True))
        if removed_cols:
            print(f"Removed low-variance columns: {list(removed_cols)}")
    
    return X_reduced, selector.get_support(indices=True)


def remove_correlated_features(X: np.ndarray, y: np.ndarray, threshold: float = 0.7,
                                  method: str = 'pearson', verbose: bool = False):
    """
    Remove highly correlated features from a NumPy array based on correlation threshold.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,) or (n_samples, n_targets).
    threshold : float, default=0.7
        Features with correlation higher than this threshold are removed.
    method : str, default='pearson'
        Correlation method: 'pearson' or 'spearman'.
    verbose : bool, default=True
        Whether to print removed column indices.

    Returns
    -------
    X_reduced : np.ndarray
        Feature matrix with highly correlated features removed.
    kept_indices : np.ndarray
        Indices of features that were kept.
    """
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
   
    n_features = X.shape[1]
    n_targets = y.shape[1]
    corr_with_target = np.zeros((n_features, n_targets))
    
    for i in range(n_features):
        xi = X[:, i]
        for t in range(n_targets):
            yt = y[:, t]
           
            # Make sure both are 1D arrays
            xi = np.asarray(xi).ravel()
            yt = np.asarray(yt).ravel()
            
            if method == 'pearson':
                corr_with_target[i, t] = pearsonr(xi, yt)[0]
            elif method == 'spearman':
                corr_with_target[i, t] = spearmanr(xi, yt)[0]
    
    # --- 2. Aggregate across targets (one score per feature)
    feature_score = np.mean(np.abs(corr_with_target), axis=1)
        
    sorted_indices = np.argsort(feature_score)[::-1]            
    X_sorted = X[:, sorted_indices]

    corr_matrix = np.corrcoef(X_sorted.T)

    drop_indices = set()
    for i in range(n_features - 1):
        for j in range(i + 1):
            val = abs(corr_matrix[j, i + 1])
            if val >= threshold:
                drop_indices.add(i + 1)

    keep_indices = [idx for idx in range(n_features) if idx not in drop_indices]
    if verbose and drop_indices:
        print(f"Removed highly correlated columns: {sorted_indices[list(drop_indices)]}")
    
    return X_sorted[:, keep_indices], sorted_indices[keep_indices]


def preprocess_features(X: np.ndarray, y: np.ndarray, var_threshold: float = 0.1,
                           corr_threshold: float = 0.7, corr_method: str = 'pearson',
                           verbose: bool = False):
    """
    Full preprocessing pipeline for NumPy feature arrays: removes low-variance
    features and highly correlated features.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    var_threshold : float, default=0.1
        Variance threshold for removing low-variance features.
    corr_threshold : float, default=0.7
        Correlation threshold for removing highly correlated features.
    corr_method : str, default='pearson'
        Correlation method for feature-target correlation: 'pearson' or 'spearman'.
    verbose : bool, default=True
        Whether to print information about removed features.

    Returns
    -------
    X_processed : np.ndarray
        Preprocessed feature matrix.
    kept_idx : np.ndarray
    Indices of original features that were kept.
    """
    print(f"Remove features that have variance below {var_threshold}")
    X_lv, lv_indices = remove_low_variance(X, threshold=var_threshold, verbose=verbose)
    print(f"Remove correlated features with a correlation coefficient greater than {corr_threshold}")
    X_corr, corr_indices = remove_correlated_features(X_lv, y, threshold=corr_threshold, method=corr_method, verbose=verbose)
    
    # Map final kept indices back to original feature indices
    kept_idx = lv_indices[corr_indices]
    
    return X_corr, kept_idx

def recursive_feature_elimination(X: np.ndarray, 
                                  y: np.ndarray,
                                  estimator=None,
                                  step: int = 1,
                                  cv_splits: int = 5,
                                  scoring: str = 'r2',
                                  n_features_to_select: int = None,
                                  verbose: bool = False,
                                  n_jobs=-1):
    """
    Perform RFECV to automatically select the optimal number of features.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target vector (n_samples,)
    estimator : sklearn BaseEstimator, default=None
        Model used to evaluate feature importance. Defaults to LinearRegression.
    step : int, default=1
        Number of features to remove at each iteration.
    cv_splits : int, default=5
        Number of cross-validation folds.
    scoring : str, default='r2'
        Scoring metric used to evaluate performance.
    n_features_to_select : int, default=None
        If provided, select this number of features manually. Otherwise, RFECV
        will automatically determine the optimal number.
    verbose : bool, default=True
        Print progress and results.
    n_jobs : int, default=-1
        Number of parallel jobs.

    Returns
    -------
    X_rfecv : np.ndarray
        Feature matrix reduced to selected features.
    selected_indices : np.ndarray
        Indices of features selected from original X.
    rfecv : sklearn.feature_selection.RFECV
        Fitted RFECV object.
    """
    if estimator is None:
        estimator = get_model('Random_Forest')

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    if n_features_to_select is None:
        # Automatic feature selection
        selector = RFECV(
            estimator=estimator,
            step=step,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs
        )
        selector.fit(X, y)
    else:
        # Manual number of features
        selector = RFE(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            step=step
        )
        selector.fit(X, y)

    selected_indices = np.where(selector.support_)[0]
    X_selected = X[:, selected_indices]

    if verbose:
        print(f"Selected feature indices: {selected_indices}")
        if n_features_to_select is None:
            print(f"Optimal number of features (RFECV): {selector.n_features_}")
            # Use cv_results_ instead of grid_scores_
            if hasattr(selector, 'cv_results_'):
                print(f"CV scores: {selector.cv_results_['mean_test_score']}")
            else:
               print("CV scores not available")
    else:
        # Evaluate manual selection with CV
        scores = cross_val_score(estimator, X_selected, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
        print(f"CV {scoring}: {scores.mean():.4f}")

    return X_selected, selected_indices, selector


#%% Example

# from sklearn.datasets import load_diabetes

# diabetes = load_diabetes()
# X, y = diabetes.data, diabetes.target

# X_processed, X_idx = preprocess_features(X, y, var_threshold=0.0, corr_threshold=0.8)
# print("Processed shape:", X_processed.shape)


# X_selected, idx_selected, selector = recursive_feature_elimination(X, y, estimator=None, n_features_to_select=None)
# print("Shape after RFE:", X_selected.shape)
    
    
    

