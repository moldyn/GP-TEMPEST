<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/hero_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/hero_light.png">
    <img alt="GP-TEMPEST" src="docs/hero_light.png" width="800">
  </picture>
</p>

# GP-TEMPEST

**Gaussian Process Temporal Embedding for Protein Simulations and Transitions**

<p align="center">
  <a href="https://doi.org/10.1063/5.0282147"><img src="https://img.shields.io/badge/DOI-10.1063%2F5.0282147-blue" alt="DOI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
  <a href=".github/workflows/pytest.yml"><img src="https://github.com/moldyn/GP-TEMPEST/actions/workflows/pytest.yml/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/moldyn/GP-TEMPEST"><img src="https://codecov.io/gh/moldyn/GP-TEMPEST/branch/main/graph/badge.svg" alt="Coverage"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch" alt="PyTorch">
  <a href="https://moldyn.github.io/GP-TEMPEST"><img src="https://img.shields.io/badge/docs-MkDocs-526CFE?logo=materialformkdocs" alt="Docs"></a>
  <a href="https://pypi.org/project/gp-tempest"><img src="https://img.shields.io/pypi/v/gp-tempest" alt="PyPI"></a>
  <a href="https://pypistats.org/packages/gp-tempest"><img src="https://img.shields.io/pypi/dt/gp-tempest" alt="Downloads"></a>
  <a href="https://moldyn.github.io/GP-TEMPEST/tutorial/tutorial/"><img src="https://img.shields.io/badge/tutorial-notebook-orange?logo=jupyter" alt="Tutorial"></a>
</p>

<p align="center">
  <a href="https://moldyn.github.io/GP-TEMPEST">Documentation</a> •
  <a href="https://moldyn.github.io/GP-TEMPEST/tutorial/tutorial/">Tutorial</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#citation">Citation</a>
</p>

---

GP-TEMPEST is a PyTorch implementation of the Gaussian Process Variational
Autoencoder (GP-VAE) framework for time-aware dimensionality reduction of
molecular dynamics (MD) simulations. The method leverages physics-informed
Gaussian Process priors to capture temporal correlations in the latent space,
enabling the recovery of hidden or kinetically relevant degrees of freedom in
complex biomolecular systems.

## Features

- **Physics-informed dimensionality reduction** using Gaussian Processes as temporal priors
- **Flexible kernel selection** with support for the Matérn kernel 
(ν = 0.5, 1.5, 2.5)
- **Sparse GP inference** with inducing points for scalability to large 
molecular trajectories
- **Recovery of hidden degrees of freedom** not accessible in any projection 
of the input data
- **Free-energy landscapes and kinetic insight** from GP-smoothed, physically interpretable latent coordinates

## Installation

```bash
pip install gp-tempest
```

> **Note:** PyTorch is listed as a dependency but pip will install the CPU version by default. For GPU support install torch manually first:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> pip install gp-tempest
> ```

**From source:**
```bash
git clone https://github.com/moldyn/GP-TEMPEST.git
cd GP-TEMPEST
pip install -e .
```

## Usage

### Command-line interface

**Fully-connected variant:**
```bash
# Generate a default config file
python tempest_main.py --generate_config

# Run with your config
python tempest_main.py --config my_config.yaml
```

### Python API

```python
import numpy as np
import torch
from gptempest import TEMPEST, MaternKernel, load_prepare_data

# Set up kernel and model
kernel = MaternKernel(scale=10.0, nu=1.5, dtype=torch.float64)
inducing_points = np.linspace(0, 1, 50)

model = TEMPEST(
    cuda=False,
    kernel=kernel,
    dim_input=dim_input,
    dim_latent=2,
    layers_hidden_encoder=[128, 64],
    layers_hidden_decoder=[64, 128],
    inducing_points=inducing_points,
    beta=1.0,
    N_data=N_data,
    dtype=torch.float64,
)

# Train
model.train_model(dataset, train_size=1.0, learning_rate=1e-3,
                  weight_decay=1e-5, batch_size=512, n_epochs=100)

# Extract latent space
embedding = model.extract_latent_space(dataset, batch_size=512)
```

### Configuration file

GP-TEMPEST is configured via YAML files. Generate a template with `--generate_config` and adjust the following key parameters.
The discussion of these parameters can be found in the paper.

| Parameter | Description |
|-----------|-------------|
| `dim_latent` | Dimensionality of the latent space (typically 2) |
| `layers_hidden` | Hidden layer sizes for encoder/decoder |
| `kernel_nu` | Matérn kernel smoothness (0.5, 1.5, or 2.5) |
| `kernel_scale` | Time-scale of the GP prior |
| `beta` | Weight of the GP regularization term |
| `inducing_points` | Path to inducing point time coordinates |

## Citation

If you use GP-TEMPEST in your research, please cite:

```bibtex
@article{diez2025gptempest,
  title   = {Recovering Hidden Degrees of Freedom Using Gaussian Processes},
  author  = {Diez, Georg and Dethloff, Nele and Stock, Gerhard},
  journal = {J. Chem. Phys.},
  volume  = {163},
  pages   = {124105},
  year    = {2025},
  doi     = {10.1063/5.0282147}
}
```

> G. Diez, N. Dethloff, G. Stock,
> "Recovering Hidden Degrees of Freedom Using Gaussian Processes,"
> *J. Chem. Phys.* **163**, 124105 (2025), https://doi.org/10.1063/5.0282147

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
