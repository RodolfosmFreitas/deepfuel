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
4. [Contributors](#contributors)

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
### From Source:
 
```bash
git clone https://github.com/RodolfosmFreitas/deepfuel.git
cd deepfuel
pip install .
# install in editable mode
pip install -e .
# or straight from git
pip install git+https://github.com/RodolfosmFreitas/deepfuel.git
```
### Dependencies

- **Core**: `numpy`, `pytorch`, `rdkit`
  
---

## Quick Start

---

## Contributors

- Rodolfo Freitas ([@RodolfosmFreitas](https://github.com/RodolfosmFreitas)) — [rodolfo.dasilvamachadodefreitas@qmul.ac.uk](mailto:rodolfo.dasilvamachadodefreitas@qmul.ac.uk)
- Zhihao Xing([@Zhihao-07](https://github.com/Zhihao-07)) — [zhihao.xing@qmul.ac.uk](mailto:zhihao.xing@qmul.ac.uk)
- Xi Jiang — [xi.jiang@qmul.ac.uk](mailto:xi.jiang@qmul.ac.uk)

## Contributing & Contact

Contributions welcome! Please open an issue or pull request and get in touch with any questions [here](https://github.com/RodolfosmFreitas/deepfuel/issues).
