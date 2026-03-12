# Installation

## Requirements

- Python 3.9+
- PyTorch 2.0+

## From GitHub

```bash
pip install git+https://github.com/moldyn/GP-TEMPEST.git
```

## From source

```bash
git clone https://github.com/moldyn/GP-TEMPEST.git
cd GP-TEMPEST
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU only
# pip install torch  # for GPU (CUDA)
pip install -e .
```

## Verify

```python
from gptempest import TEMPEST, MaternKernel
print("GP-TEMPEST installed successfully")
```
