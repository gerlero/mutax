import multiprocessing
from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from mutax import differential_evolution

jax.config.update("jax_num_cpu_devices", multiprocessing.cpu_count())


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


@pytest.mark.parametrize("strategy", ["rand1bin", "best1bin"])
@pytest.mark.parametrize("updating", ["immediate", "deferred"])
@pytest.mark.parametrize("workers", [1, 2, -1])
@pytest.mark.parametrize("x0", [None, [0.0, 0.0]])
@pytest.mark.parametrize("polish", [True, False])
def test_differential_evolution(
    *,
    strategy: Literal["rand1bin", "best1bin"],
    updating: Literal["immediate", "deferred"],
    workers: int,
    x0: jax.Array | None,
    polish: bool,
) -> None:
    bounds = jnp.array([[-5.0, 5.0], [-5.0, 5.0]])
    result = differential_evolution(
        rosenbrock,
        bounds,
        key=jax.random.key(0),
        strategy=strategy,
        updating=updating,
        workers=workers,
        x0=x0,
        polish=polish,
    )
    assert result.success
    assert result.status == 0
    assert result.x == pytest.approx([1.0, 1.0])
    assert result.fun == pytest.approx(0.0)
    assert result.nit < 200
