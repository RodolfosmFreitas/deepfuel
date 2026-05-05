
"""
ML tools to construct QSPR models
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from deepfuel.pytorch_models import DKLRegressor, DeepNet, CADGM, BayesianNNRegressor
from deepfuel.pyg_models import GNNRegressor
#%% Get the models
def get_model(name: str, **kwargs):
    """
    Parameters
    ----------
    name : str
        Name of the model to be built: 
    **kwargs : TYPE
        keyword arguments of the model.

    Raises
    ------
    ValueError
        Choose the a model that is not available.

    Returns
    -------
    TYPE
        Return a ML model instance by name.
    """
    models = {
        "Linear": LinearRegression,
        "Ridge": Ridge,
        "Lasso": Lasso,
        "Linear_SVR": LinearSVR,
        "MLP": MLPRegressor,
        "Random_Forest": RandomForestRegressor,
        "Extra_Trees_Regressor": ExtraTreesRegressor,
        "GBR": GradientBoostingRegressor,
        "Decision_Tree": DecisionTreeRegressor,
        "GP": GaussianProcessRegressor,
        "DKL": DKLRegressor,
        "XGB": XGBRegressor,
        "LGBM":LGBMRegressor,
        "DeepNN": DeepNet,
        "CADGM": CADGM,
        "BayesNN": BayesianNNRegressor,
        "GNN": GNNRegressor,
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Available: {list(models.keys())}")
    return models[name](**kwargs)


