<p align="center">
  <img width="8000" height="4041" alt="DeepFuel" src="https://github.com/user-attachments/assets/e6a55d06-e784-4a8a-a553-da1d19939884" />
</p>

# DeepFuel-kit: AI-Guided Formulation of Next-Generation Sustainable Fuels

**DeepFuel-kit** is an open-source Python library for AI-guided formulation of next-generation sustainable fuels.


[![License](https://img.shields.io/github/license/RodolfosmFreitas/deepfuel)](https://github.com/RodolfosmFreitas/deepfuel/blob/main/LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![PyPI](https://img.shields.io/pypi/v/deepfuel)](https://pypi.org/project/deepfuel/)
[![TestPyPI](https://img.shields.io/badge/TestPyPI-available-blue)](https://test.pypi.org/project/deepfuel/)
[![Downloads](https://static.pepy.tech/badge/deepfuel)](https://pepy.tech/project/deepfuel)
[![Powered by: uv](https://img.shields.io/badge/-uv-purple)](https://docs.astral.sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/RodolfosmFreitas/deepfuel/ci.yml?branch=main&logo=github-actions)](https://github.com/RodolfosmFreitas/deepfuel/actions)
[![codecov](https://codecov.io/gh/RodolfosmFreitas/deepfuel/branch/main/graph/badge.svg)](https://codecov.io/gh/RodolfosmFreitas/deepfuel)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20045282.svg)](https://doi.org/10.5281/zenodo.20045282)

DeepFuel-kit empowers researchers and innovators to utilise advanced AI, molecular feature extraction, and predictive modelling tools, thereby promoting the efficient, insightful, and environmentally sustainable design of next-generation fuels. The application of DeepFuel includes (but is not limited to):
1. **“Drop-in” alternative fuel & fuel additive design**
   - Formulation of “drop-in” fuel mixtures (such as SAF): This involves designing fuels that meet performance and regulatory standards and are fully compatible with existing infrastructure and engines without requiring modifications. Using components produced from renewable feedstocks, the designed fuel blends reduce lifecycle greenhouse gas (GHG) emissions. It might involve selecting or developing new fuel components or molecules.
   - Development of fuel additives for blends with user-defined properties: This involves selection or development of new fuel components or molecules, and adding them to existing fuels (including conventional fossil fuels), achieving specific outcomes such as targeted physicochemical properties (e.g. lower freezing point & viscosity) for enhanced performance.
2. **Low-carbon multi-fuel design**
   - Formulation of low-carbon fuel mixtures based on existing low- or zero-carbon fuels: This involves combining existing low or zero-carbon fuels (e.g. ammonia, biodiesel, dimethyl ether, methanol) to produce a final multi-fuel product that offers optimal combustion and reduces lifecycle GHG emissions without requiring new components/molecules. New technologies may be needed to burn the formulated multi-fuel.

## Table of Contents

1. [Key Capabilities](#key-capabilities)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Developers](#developers)
5. [Contributing & Contact](#contributing--contact)

---

## Key Capabilities
- **Molecular Featurization at Scale**: High-throughput molecular featurization pipelines for classical descriptors and deep learning embeddings, enabling large-scale fuel property prediction.
- **Predictive & Generative Modelling**: State-of-the-art predictive models for property estimation and generative frameworks for designing novel sustainable fuel candidates.
- **Inverse Fuel Design**: Advanced inverse design strategies leveraging diffusion models, genetic algorithms, SciPy optimisation, and reinforcement learning to explore chemical space and optimise fuel formulations.
- **Uncertainty Quantification**: Integration of methods like EnbPI for robust prediction intervals to ensure reliable surrogate model predictions.
- **Simulation Acceleration**: ML surrogates to accelerate 0D and multidimensional combustion simulations while maintaining accuracy.
- **Data Integration & Visualisation**: Tools for combining experimental, simulation, and computational data with interactive visualisations to explore fuel performance efficiently.

---

## Installation

### From PyPI

```bash
pip install deepfuel
```

### From Source

```bash
git clone https://github.com/RodolfosmFreitas/deepfuel.git
cd deepfuel
pip install .

# Install in editable mode
pip install -e .

# Or install directly from GitHub
pip install git+https://github.com/RodolfosmFreitas/deepfuel.git
```

### Dependencies

DeepFuel relies on the following core scientific computing, machine learning, and uncertainty quantification libraries:

* **Scientific Computing**: `numpy`, `pandas`, `scipy`
* **Machine Learning**: `scikit-learn`, `xgboost`, `lightgbm`
* **Deep Learning**: `torch`, `gpytorch`, `torch-geometric`, `transformers`
* **Chemoinformatics**: `deepchem`
* **Molecular Embeddings**: `gensim`
* **Uncertainty Quantification**: `mapie`
* **Sensitivity Analysis**: `SALib`
* **Optimization**: `optuna`, `pymoo`
* **Reinforcement Learning**: `gymnasium`, `stable-baselines3`, `shimmy`
* **Experiment Tracking**: `wandb`
* **Utilities**: `joblib`, `threadpoolctl`, `imageio`

All dependencies are installed automatically when installing DeepFuel via `pip`.

## Quick Start

DeepFuel provides tools for fuel property prediction, inverse fuel design, and multi-objective fuel formulation.

### 1. Property Prediction

Train a graph neural network (GNN) to predict a fuel property directly from molecular SMILES strings.

```python
import pandas as pd
from deepfuel.data_utils import prepare_data
from deepfuel.models import get_model

# Load dataset
data = pd.read_excel("data-fuel-properties.xlsx")

# Select target property
X = data["SMILES"].tolist()
y = data["YSI"].values.reshape(-1, 1)

# Split and preprocess
X_train, X_test, y_train, y_test, scalerX, scalery = prepare_data(
    X,
    y,
    train_size=0.8,
    scaler_X=None,
    scaler_y="mad",
)

# Create model
model = get_model(
    "GNN",
    conv_layer="MFConv",
    epochs=500,
    batch_size=64,
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = scalery.inverse_transform(model.predict(X_test))

print(y_pred[:5])
```

#### Hyperparameter Optimisation

DeepFuel provides Optuna-based hyperparameter optimisation:

```python
from deepfuel.hpo import tune

study, best_model = tune(
    model,
    suggest_params,
    X_train,
    y_train,
    metric="r2",
    n_splits=5,
    n_trials=100,
)
```

### Supported Fuel Properties

DeepFuel currently supports modelling of:

* Boiling Point (`BP`)
* Density (`rho`)
* Lower Heating Value (`LHV`)
* Freezing Point (`FP`)
* Surface Tension (`ST`)
* Threshold Sooting Index (`TF`)
* Yield Sooting Index (`YSI`)
* Derived Cetane Number (`DCN`)

---

### 2. Inverse Fuel Design

Design a sustainable aviation fuel (SAF) blend that reproduces target fuel properties while satisfying regulatory constraints.

```python
import numpy as np
import pandas as pd

from deepfuel.io import load_model
from deepfuel.mol_featurizer import featurize_molecules
from deepfuel.scipy_optimizers import optimize_fuel

# Load candidate molecules
data = pd.read_excel("SAF-palette.xlsx")
smiles = data["SMILES"].tolist()

# Generate molecular embeddings
phi = featurize_molecules(
    featurizer="Mol2VecFingerprint",
    smiles_list=smiles,
)

# Load pre-trained surrogate models
model = load_model("Models/MLP-predictor-rho-LHV-FP/best_model.joblib")

# Target Jet-A properties
target = np.array([
    0.806,  # Density
    42.8,   # Lower Heating Value
    47.0    # Flash Point
])

# Perform inverse design
best_x, best_loss, solutions = optimize_fuel(
    objective_function,
    args=(phi, target),
    optimizer="SLSQP",
    n_features=len(smiles),
    n_starts=1000,
)

print("Optimal blend composition:")
print(best_x)
```

---

### 3. Multi-Objective Fuel Formulation

Optimise fuel blends using hybrid Genetic Algorithm and Reinforcement Learning strategies while satisfying fuel-property constraints.

```python
import pandas as pd

from deepfuel.io import load_model
from deepfuel.mol_featurizer import featurize_molecules
from deepfuel.hybrid_optim import HybridOptimization

# Load candidate fuel molecules
data = pd.read_excel("Diesel-palette.xlsx")
smiles = data["SMILES"].tolist()

# Generate molecular representations
phi = featurize_molecules(
    featurizer="Mol2VecFingerprint",
    smiles_list=smiles,
)

# Load surrogate models
model_dcn = load_model("Models/MLP-predictor-DCN/best_model.joblib")
model_ysi = load_model("Models/MLP-predictor-YSI/best_model.joblib")

# Define optimisation objectives
def objective_function(x):
    blend = x[None, :] @ phi

    dcn = model_dcn.predict(blend)
    ysi = model_ysi.predict(blend)

    return [
        -dcn,   # Maximise cetane number
         ysi    # Minimise soot emissions
    ]

# Hybrid optimization
optimizer = HybridOptimization(
    n_comp=len(smiles),
    obj_fun=objective_function,
    n_obj=2,
    n_cycles=3,
)

result, agent = optimizer.run()

print("Pareto-optimal fuel formulations:")
print(result.X)
```

---

### Typical Applications

* Sustainable Aviation Fuel (SAF) formulation
* Renewable diesel (HVO) design
* Fuel additive discovery
* Low-soot fuel optimization
* Multi-fuel blending
* Surrogate-assisted combustion modeling
* Molecular property prediction
* Inverse molecular and fuel design

### Supported Optimisation Methods

* Gradient-based optimisation (SLSQP, trust-constr)
* Genetic Algorithms (NSGA-II & NSGA-III)
* Hybrid Genetic Algorithm + Reinforcement Learning
* Multi-objective Pareto optimisation
* Constraint-aware fuel formulation
* Hyperparameter optimisation with Optuna
* Uncertainty quantification with MAPIE and EnbPI


---

## Developers

- Rodolfo Freitas ([@RodolfosmFreitas](https://github.com/RodolfosmFreitas)) — [rodolfo.dasilvamachadodefreitas@qmul.ac.uk](mailto:rodolfo.dasilvamachadodefreitas@qmul.ac.uk)
- Zhihao Xing([@Zhihao-07](https://github.com/Zhihao-07)) — [zhihao.xing@qmul.ac.uk](mailto:zhihao.xing@qmul.ac.uk)
- Xi Jiang — [xi.jiang@qmul.ac.uk](mailto:xi.jiang@qmul.ac.uk)

---

## Contributing & Contact

Contributions are welcome! Please open an issue or submit a pull request [here](https://github.com/RodolfosmFreitas/deepfuel/issues). 

For questions, suggestions, or collaboration opportunities, please contact Rodolfo Freitas at <rodolfo.dasilvamachadodefreitas@qmul.ac.uk>.

See the repository issue tracker for bug reports, feature requests, and discussions.
