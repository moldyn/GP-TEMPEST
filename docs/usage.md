# Usage

## Command-line interface

Generate a default config file and run training:

```bash
# Generate template config
tempest --generate_config

# Train with your config
tempest --config my_config.yaml
```

## Configuration parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data_path` | Path to input coordinates file (whitespace-separated) | — |
| `inducing_points_path` | Path to inducing point timestamps | — |
| `save_path` | Output directory | — |
| `dim_input` | Number of input features | — |
| `dim_latent` | Latent space dimensionality | 2 |
| `neurons_ae` | Hidden layer sizes, e.g. `[32, 32, 32]` | `[32, 32, 32]` |
| `epochs` | Training epochs | 100 |
| `batch_size` | Batch size (larger is better, ≥512 recommended) | 1024 |
| `learning_rate` | AdamW learning rate | 1e-4 |
| `weight_decay` | AdamW weight decay | 1e-6 |
| `beta` | Weight of the GP regularization term | 50 |
| `kernel_nu` | Matérn smoothness: 0.5, 1.5, or 2.5 | 1.5 |
| `kernel_scale` | Time scale of the GP prior | 1000 |
| `cuda` | Use GPU if available | true |

## Python API

```python
import numpy as np
import torch
from gptempest import TEMPEST, MaternKernel
from gptempest.utils import load_prepare_data

# Load data
dataset         = load_prepare_data("data.dat", dtype=torch.float64)
inducing_points = np.loadtxt("inducing_points.dat")

# Build model
kernel = MaternKernel(nu=1.5, scale=1e3, dtype=torch.float64)

model = TEMPEST(
    cuda=torch.cuda.is_available(),
    kernel=kernel,
    dim_input=2,
    dim_latent=2,
    layers_hidden_encoder=[32, 32, 32],
    layers_hidden_decoder=[32, 32, 32],
    inducing_points=inducing_points,
    beta=50.0,
    N_data=len(dataset),
    dtype=torch.float64,
)

# Train
model.train_model(
    dataset,
    train_size=1,
    learning_rate=1e-4,
    weight_decay=1e-6,
    batch_size=1024,
    n_epochs=100,
)

# Extract embedding
embedding = model.extract_latent_space(dataset, batch_size=1024)
np.savetxt("embedding.dat", embedding, fmt="%.4f")
```

## Choosing inducing points

Inducing points are timestamps that should cover the important events in your trajectory — metastable states and transitions. A simple choice is to use uniformly spaced time points:

```python
n_inducing = 50
inducing_points = np.linspace(0, len(data) - 1, n_inducing)
np.savetxt("inducing_points.dat", inducing_points)
```

For best results, choose points that sample transitions and metastable regions.

## Kernel smoothness (ν)

| ν | Process | When to use |
|---|---------|-------------|
| 0.5 | Ornstein–Uhlenbeck | Rough, fast dynamics |
| 1.5 | Once differentiable | General MD trajectories |
| 2.5 | Twice differentiable | Smoother, slower dynamics |
