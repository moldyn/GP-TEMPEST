<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="hero_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="hero_light.png">
    <img alt="GP-TEMPEST" src="hero_light.png" width="700">
  </picture>
</p>

# GP-TEMPEST

**Gaussian Process Temporal Embedding for Protein Simulations and Transitions**

GP-TEMPEST is a PyTorch implementation of a Gaussian Process Variational Autoencoder (GP-VAE) for time-aware dimensionality reduction of molecular dynamics (MD) simulations. The method uses physics-informed Gaussian Process priors to capture temporal correlations in the latent space, enabling the recovery of hidden or kinetically relevant degrees of freedom in complex biomolecular systems.

## Features

- **Physics-informed dimensionality reduction** using Gaussian Processes as temporal priors
- **Flexible kernel selection** with support for the Matérn kernel (ν = 0.5, 1.5, 2.5)
- **Sparse GP inference** with inducing points for scalability to large MD trajectories
- **Simple Python API** — train and embed in a few lines of code

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

> G. Diez, N. Dethloff, G. Stock, "Recovering Hidden Degrees of Freedom Using Gaussian Processes," *J. Chem. Phys.* **163**, 124105 (2025), [https://doi.org/10.1063/5.0282147](https://doi.org/10.1063/5.0282147)
