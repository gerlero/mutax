import multiprocessing
from collections.abc import Callable
from typing import Literal

import equinox as eqx
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


# Asymmetric and per-dimension different, so that swapping the two bounds or using
# only one dimension's bounds does not go unnoticed
X0_BOUNDS = jnp.array([[-1.0, 3.0], [-2.0, 1.0]])


def sphere(x: jax.Array) -> jax.Array:
    return jnp.sum(x**2, axis=0)


@pytest.mark.parametrize("strategy", ["rand1bin", "best1bin"])
@pytest.mark.parametrize("updating", ["immediate", "deferred"])
@pytest.mark.parametrize("polish", [True, False])
@pytest.mark.parametrize(
    "x0",
    [
        [-5.0, 0.0],  # below the lower bound
        [0.0, 5.0],  # above the upper bound
        [-5.0, 5.0],  # both, one in each direction
        [10.0, -10.0],  # both, the other way around
        [0.0, -10.0],  # below the lower bound of the second dimension only
    ],
)
def test_x0_out_of_bounds(
    *,
    strategy: Literal["rand1bin", "best1bin"],
    updating: Literal["immediate", "deferred"],
    polish: bool,
    x0: list[float],
) -> None:
    with pytest.raises(eqx.EquinoxRuntimeError, match="x0 lay outside"):
        differential_evolution(
            sphere,
            X0_BOUNDS,
            key=jax.random.key(0),
            strategy=strategy,
            updating=updating,
            polish=polish,
            x0=jnp.array(x0),
        )


@pytest.mark.parametrize(
    "x0",
    [
        [0.0, 0.0],
        [-1.0, 1.0],  # exactly on the bounds, which is allowed
        [3.0, -2.0],  # the other two, also exactly on the bounds
        [2.5, -1.5],
    ],
)
def test_x0_in_bounds(*, x0: list[float]) -> None:
    target = jnp.array(x0)

    def cost(x: jax.Array) -> jax.Array:
        return jnp.sum((x - target) ** 2)

    # No generations are run, so the result can only be the best member of the
    # initial population: this pins that `x0` is in it, and unmodified
    result = differential_evolution(
        cost, X0_BOUNDS, key=jax.random.key(0), maxiter=0, polish=False, x0=target
    )
    assert result.x == pytest.approx(x0)


def test_x0_out_of_bounds_traced() -> None:
    # `x0` is a tracer here, so the check cannot be a plain Python `if`
    @eqx.filter_jit
    def run(x0: jax.Array) -> jax.Array:
        return differential_evolution(
            sphere, X0_BOUNDS, key=jax.random.key(0), polish=False, x0=x0
        ).x

    assert jnp.allclose(run(jnp.array([0.0, 0.0])), 0.0, atol=1e-5)
    with pytest.raises(eqx.EquinoxRuntimeError, match="x0 lay outside"):
        run(jnp.array([0.0, 5.0]))


def test_x0_out_of_bounds_vmapped() -> None:
    run = jax.vmap(
        lambda x0: (
            differential_evolution(
                sphere, X0_BOUNDS, key=jax.random.key(0), polish=False, x0=x0
            ).x
        )
    )

    assert jnp.allclose(run(jnp.array([[0.0, 0.0], [1.0, -1.0]])), 0.0, atol=1e-5)
    # Only one of the two batch elements is out of bounds
    with pytest.raises(eqx.EquinoxRuntimeError, match="x0 lay outside"):
        run(jnp.array([[0.0, 0.0], [0.0, 5.0]]))
    with pytest.raises(eqx.EquinoxRuntimeError, match="x0 lay outside"):
        run(jnp.array([[0.0, 5.0], [0.0, 0.0]]))


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
