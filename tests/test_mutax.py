import multiprocessing
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from parajax import parallelize

from mutax import differential_evolution

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
def test_differential_evolution(
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


@pytest.mark.parametrize("strategy", ["rand1bin", "best1bin"])
@pytest.mark.parametrize("updating", ["immediate", "deferred"])
@pytest.mark.parametrize("polish", [True, False])
@pytest.mark.parametrize(
    "x0", [None, [-5.0, 5.0], [-5.0, 0.0], [0.0, 5.0], [-1.0, 1.0]]
)
def test_x0_out_of_bounds(
    *,
    strategy: Literal["rand1bin", "best1bin"],
    updating: Literal["immediate", "deferred"],
    polish: bool,
    x0: jax.Array | None,
) -> None:
    # Objective minimized at [-5, 5], which lies outside `bounds` past the lower
    # bound in one dimension and past the upper bound in the other, so the
    # constrained minimum is the corner [-1, 1]
    bounds = jnp.array([[-1.0, 3.0], [-2.0, 1.0]])

    def cost(x: jax.Array) -> jax.Array:
        return jnp.sum((x - jnp.array([-5.0, 5.0])) ** 2)

    result = differential_evolution(
        cost,
        bounds,
        key=jax.random.key(0),
        strategy=strategy,
        updating=updating,
        polish=polish,
        x0=x0,
    )
    assert jnp.all(result.x >= bounds[:, 0])
    assert jnp.all(result.x <= bounds[:, 1])
    assert result.x == pytest.approx([-1.0, 1.0])


@pytest.mark.parametrize(
    ("x0", "clipped"),
    [
        ([-5.0, 5.0], [-1.0, 1.0]),
        ([-5.0, 0.0], [-1.0, 0.0]),
        ([0.0, 5.0], [0.0, 1.0]),
        ([10.0, -10.0], [3.0, -2.0]),
        ([-1.0, 1.0], [-1.0, 1.0]),
        ([0.0, 0.0], [0.0, 0.0]),
    ],
)
def test_x0_clipped_into_initial_population(
    *, x0: jax.Array, clipped: jax.Array
) -> None:
    bounds = jnp.array([[-1.0, 3.0], [-2.0, 1.0]])
    target = jnp.array(x0)

    def cost(x: jax.Array) -> jax.Array:
        return jnp.sum((x - target) ** 2)

    # No generations are run, so the result is the best member of the initial
    # population, and the clipped guess is the point of `bounds` closest to `target`
    result = differential_evolution(
        cost, bounds, key=jax.random.key(0), maxiter=0, polish=False, x0=x0
    )
    assert result.x == pytest.approx(clipped)


def test_invalid() -> None:
    bounds = jnp.array([[-5.0, 5.0], [-5.0, 5.0]])
    with pytest.raises(ValueError, match="strategy"):
        differential_evolution(rosenbrock, bounds, strategy="invalid")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="updating"):
        differential_evolution(rosenbrock, bounds, updating="invalid")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="workers"):
        differential_evolution(rosenbrock, bounds, workers=-2)
    with pytest.raises(ValueError, match="vectorized"):
        differential_evolution(rosenbrock, bounds, vectorized=True, workers=pmap)
