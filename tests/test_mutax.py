import multiprocessing
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from mutax import differential_evolution
from parajax import parallelize

jax.config.update("jax_num_cpu_devices", multiprocessing.cpu_count())


def rosenbrock(x: jax.Array) -> jax.Array:
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0, axis=0)


def test_rosenbrock() -> None:
    x0 = jnp.array([1.0, 1.0])
    assert rosenbrock(x0) == 0.0
    x1 = jnp.array([0.0, 0.0])
    assert rosenbrock(x1) == 1.0
    x2 = jnp.array([-1.0, 1.0])
    assert rosenbrock(x2) == 4.0
    x3 = jnp.array([1.0, 2.0])
    assert rosenbrock(x3) == 100.0


def pmap(func: Callable[[jax.Array], jax.Array], x: jax.Array) -> jax.Array:
    return parallelize(jax.vmap(func))(x)


@pytest.mark.parametrize("strategy", ["rand1bin", "best1bin"])
@pytest.mark.parametrize("updating", ["immediate", "deferred"])
@pytest.mark.parametrize("workers", [1, 2, -1, pmap])
@pytest.mark.parametrize("x0", [None, [0.0, 0.0]])
@pytest.mark.parametrize("polish", [True, False])
@pytest.mark.parametrize("vectorized", [False, True])
def test_differential_evolution(  # noqa: PLR0913
    *,
    strategy: Literal["rand1bin", "best1bin"],
    updating: Literal["immediate", "deferred"],
    workers: int | Callable[[Callable[[jax.Array], jax.Array], jax.Array], jax.Array],
    x0: jax.Array | None,
    polish: bool,
    vectorized: bool,
) -> None:
    if callable(workers) and vectorized:
        pytest.skip("Cannot use callable workers with vectorized=True")

    bounds = jnp.array([[-5.0, 5.0], [-5.0, 5.0]])
    result = differential_evolution(
        rosenbrock,
        bounds,
        strategy=strategy,
        updating=updating,
        workers=workers,
        x0=x0,
        polish=polish,
        vectorized=vectorized,
    )
    assert result.success
    assert result.status == 0
    assert result.x == pytest.approx([1.0, 1.0])
    assert result.fun == pytest.approx(0.0)
    assert result.nit < 200


@pytest.mark.parametrize("polish", [True, False])
def test_workers_same_result(*, polish: bool) -> None:
    bounds = jnp.array([[-5.0, 5.0], [-5.0, 5.0]])
    result = differential_evolution(
        rosenbrock,
        bounds,
        key=jax.random.key(42),
        polish=polish,
        updating="deferred",
    )
    result2 = differential_evolution(
        rosenbrock,
        bounds,
        key=jax.random.key(42),
        polish=polish,
        workers=2,
    )
    result3 = differential_evolution(
        rosenbrock,
        bounds,
        key=jax.random.key(42),
        polish=polish,
        workers=-1,
    )
    assert result.success
    assert result2.success
    assert result3.success
    assert jnp.all(result2.x == result.x)
    assert jnp.all(result3.x == result.x)


def test_invalid() -> None:
    bounds = jnp.array([[-5.0, 5.0], [-5.0, 5.0]])
    with pytest.raises(ValueError, match="strategy"):
        differential_evolution(rosenbrock, bounds, strategy="invalid")
    with pytest.raises(ValueError, match="updating"):
        differential_evolution(rosenbrock, bounds, updating="invalid")
    with pytest.raises(ValueError, match="workers"):
        differential_evolution(rosenbrock, bounds, workers=-2)
    with pytest.raises(ValueError, match="vectorized"):
        differential_evolution(rosenbrock, bounds, vectorized=True, workers=pmap)
