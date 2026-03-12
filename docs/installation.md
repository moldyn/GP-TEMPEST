# Installation

## Requirements

- Python 3.9+
- PyTorch 2.0+

## From PyPI

```bash
pip install gp-tempest
```

## GPU support

PyTorch will default to CPU. For GPU training, install torch with CUDA first:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install gp-tempest

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install gp-tempest
```

## From source

```bash
git clone https://github.com/moldyn/GP-TEMPEST.git
cd GP-TEMPEST
pip install -e .
```

## Verify

```python
from gptempest import TEMPEST, MaternKernel
print("GP-TEMPEST installed successfully")
```
