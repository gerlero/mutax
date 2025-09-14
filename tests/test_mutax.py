from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from mutax import differential_evolution


def rosenbrock(x: jax.Array) -> jax.Array:
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def test_rosenbrock() -> None:
    x0 = jnp.array([1.0, 1.0])
    assert rosenbrock(x0) == 0.0
    x1 = jnp.array([0.0, 0.0])
    assert rosenbrock(x1) == 1.0
    x2 = jnp.array([-1.0, 1.0])
    assert rosenbrock(x2) == 4.0
    x3 = jnp.array([1.0, 2.0])
    assert rosenbrock(x3) == 100.0


@pytest.mark.parametrize("updating", ["immediate", "deferred"])
def test_differential_evolution(updating: Literal["immediate", "deferred"]) -> None:
    bounds = jnp.array([[-5.0, 5.0], [-5.0, 5.0]])
    result = differential_evolution(
        rosenbrock, bounds, key=jax.random.key(0), updating=updating
    )
    assert result.success
    assert result.status == 0
    assert result.x == pytest.approx([1.0, 1.0])
    assert result.fun == pytest.approx(0.0)
