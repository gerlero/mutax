# <div align="center">Mutax</div>

**[SciPy-like](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html) differential evolution for [JAX](https://github.com/jax-ml/jax)**

Fully [`jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html#jax.jit)ted optimization of any JAX-compatible function. Serial and parallel execution on CPU, GPU, and TPU.

[![Documentation](https://img.shields.io/readthedocs/mutax)](https://mutax.readthedocs.io/)
[![CI](https://github.com/gerlero/mutax/actions/workflows/ci.yml/badge.svg)](https://github.com/gerlero/mutax/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/gerlero/mutax/branch/main/graph/badge.svg)](https://codecov.io/gh/gerlero/mutax)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Publish](https://github.com/gerlero/mutax/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/gerlero/mutax/actions/workflows/pypi-publish.yml)
[![PyPI](https://img.shields.io/pypi/v/mutax)](https://pypi.org/project/mutax/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mutax)](https://pypi.org/project/mutax/)


## Installation

```bash
pip install mutax
```

## Quick start

```python
import jax.numpy as jnp
from mutax import differential_evolution

def cost_function(x):
    return jnp.sum(x**2)

bounds = [(-5, 5)] * 10  # 10-dimensional problem with bounds for each dimension

result = differential_evolution(cost_function, bounds)
print("Best solution:", result.x)
print("Objective value:", result.fun)
```

## Documentation

The documentation is available at [Read the Docs](https://mutax.readthedocs.io/).
