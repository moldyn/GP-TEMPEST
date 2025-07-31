# GP-TEMPEST

**Gaussian Process Temporal Embedding for Protein Simulations and Transitions**

GP-TEMPEST is a PyTorch implementation of the Gaussian Process Variational Autoencoder (GP-VAE) framework for time-aware dimensionality reduction of molecular dynamics (MD) simulations. The method leverages physics-informed Gaussian Process priors to capture temporal correlations in the latent space, enabling the recovery of hidden or kinetically relevant degrees of freedom in complex biomolecular systems.

## Features

- **Physics-informed dimensionality reduction** using Gaussian Processes as temporal priors  
- **Flexible kernel selection**, with default support for the Matérn kernel  
- **Sparse GP inference** with inducing points for scalability to large molecular trajectories  
- **Compatible with large MD datasets** and batch-wise training  

## Reference

If you use GP-TEMPEST in your research or teaching, **please cite** the following paper:

> G. Diez, N. Dethloff, G. Stock,  
> "Recovering Hidden Degrees of Freedom Using Gaussian Processes,"  
> (2025). https://arxiv.org/abs/2505.18072
