"""Evolutionary optimization algorithms in JAX."""

import warnings
from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.optimize
from jax.sharding import PartitionSpec as P

OptimizeResults = jax.scipy.optimize.OptimizeResults
"""Object holding optimization results.

**Attributes:**

- `x`: final solution.
- `success`: ``True`` if optimization succeeded.
- `status`: integer solver specific return code. 0 means converged (nominal),
  1=max BFGS iters reached, 3=zoom failed, 4=saddle point reached,
  5=max line search iters reached, -1=undefined
- `fun`: final function value.
- `nfev`: integer number of function calls used.
- `nit`: integer number of iterations of the optimization algorithm.
"""


@eqx.filter_jit
def differential_evolution(  # noqa: C901, PLR0913, PLR0915
    func: Callable[[jax.Array], jax.Array],
    /,
    bounds: jax.Array,
    *,
    key: jax.Array,
    maxiter: int = 1_000,
    popsize: int = 15,
    tol: float = 0.01,
    atol: float = 0,
    mutation: float | tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.8,
    updating: Literal["immediate", "deferred"] = "immediate",
    workers: int = 1,
    x0: jax.Array | None = None,
) -> OptimizeResults:
    """Find the global minimum of a multivariate function.

    Uses the Differential Evolution algorithm to find the global minimum of the
    given objective function within the specified bounds.

    **Arguments:**

    - `func`: The objective function to be minimized. It must take a single argument
    (a 1D array) and return a scalar.
    - `bounds`: A 2D array specifying the lower and upper bounds for each dimension of
    the input space.
    - `key`: A JAX random key for stochastic operations.
    - `maxiter`: The maximum number of generations to evolve the population.
    - `popsize`: Multiplier for setting the total population size. The population size
    is determined by `popsize * dim`.
    - `tol`: Relative tolerance for convergence.
    - `atol`: Absolute tolerance for convergence.
    - `mutation`: A float or a tuple of two floats specifying the mutation factor. If a
    tuple is provided, the mutation factor is sampled uniformly from this range for each
    mutation.
    - `recombination`: A float in [0, 1] specifying the recombination probability.
    - `updating`: Strategy for updating the population. Can be either "immediate" or
    "deferred". "immediate" updates individuals as soon as a better trial vector is
    found, while "deferred" updates the population after all trial vectors have been
    evaluated.
    - `workers`: Number of parallel workers to use for evaluating the objective
    function. If set to -1, uses all available JAX devices.
    - `x0`: Optional initial guess.

    **Returns:**
    An `OptimizeResults` object containing the optimization results.

    **Reference:**

    R. Storn and K. Price, “Differential Evolution - A Simple and Efficient Heuristic
    for global Optimization over Continuous Spaces,” Journal of Global Optimization,
    vol. 11, no. 4, pp. 341-359, Dec. 1997, doi: 10.1023/a:1008202821328.
    """
    dim = len(bounds)
    lower = jnp.array([b[0] for b in bounds])
    upper = jnp.array([b[1] for b in bounds])
    popsize *= dim

    if workers < 1 and workers != -1:
        msg = "workers must be a positive integer or -1"
        raise ValueError(msg)
    if workers == -1:
        workers = jax.local_device_count()
        if workers == 1:
            msg = (
                "differential_evolution: workers was set to -1 (use all devices), but "
                "only a single JAX device was found.\nIf running on CPU, see the "
                "mutax documentation for how to enable parallelism."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
    elif workers > jax.local_device_count():
        msg = (
            f"workers was set to {workers}, but only {jax.local_device_count()}"
            " JAX devices exist"
        )
        raise ValueError(msg)

    if workers > 1 and updating == "immediate":
        msg = (
            "differential_evolution: the 'workers' keyword has overridden "
            "updating='immediate' to updating='deferred'"
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        updating = "deferred"

    if workers > 1:
        if popsize < workers:
            workers = popsize
        else:
            popsize += (workers - popsize % workers) % workers

        mesh = jax.make_mesh((workers,), ("d",))

        def pmapped_func(x: jax.Array) -> jax.Array:
            return jax.shard_map(
                lambda x: jax.vmap(func)(x.reshape(-1, dim)),
                mesh=mesh,
                in_specs=P("d"),
                out_specs=P("d"),
                check_vma=False,
            )(x.flatten()).reshape(popsize)

    key, subkey = jax.random.split(key)
    pop = jax.random.uniform(subkey, (popsize, dim), minval=lower, maxval=upper)
    if x0 is not None:
        pop = pop.at[0].set(jnp.asarray(x0))

    fitness = pmapped_func(pop) if workers > 1 else jax.vmap(func)(pop)

    def make_trial(
        pop: jax.Array, i: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        key, subkey = jax.random.split(key)

        # Select three distinct indices from 0..pop_size-1 excluding i
        idxs = jnp.arange(popsize)
        idxs = jnp.where(idxs == i, popsize, idxs)
        idx_perm = jax.random.permutation(subkey, idxs)
        r1, r2, r3 = idx_perm[:3]
        r1 = jnp.where(r1 == popsize, idx_perm[3], r1)
        r2 = jnp.where(r2 == popsize, idx_perm[4], r2)
        r3 = jnp.where(r3 == popsize, idx_perm[5], r3)

        # Mutation
        try:
            mut_lower, mut_upper = mutation  # ty: ignore[not-iterable]
        except TypeError:
            mut_val = mutation
        else:
            key, subkey = jax.random.split(key)
            mut_val = jax.random.uniform(subkey, (), minval=mut_lower, maxval=mut_upper)

        mutant = pop[r1] + mut_val * (pop[r2] - pop[r3])
        mutant = jnp.clip(mutant, lower, upper)

        # Crossover
        key, subkey = jax.random.split(key)
        cross_points = jax.random.uniform(subkey, (dim,)) < recombination
        key, subkey = jax.random.split(key)
        cross_points = cross_points.at[jax.random.randint(subkey, (), 0, dim)].set(True)
        trial = jnp.where(cross_points, mutant, pop[i])

        return trial, key

    if updating == "immediate":

        def evolve(
            pop: jax.Array, fitness: jax.Array, key: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            def step(
                i: int, carry: tuple[jax.Array, jax.Array, jax.Array]
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                pop, fitness, key = carry
                trial, key = make_trial(pop, i, key)

                # Selection
                f_trial = func(trial)
                better = f_trial < fitness[i]
                pop = pop.at[i].set(jnp.where(better, trial, pop[i]))
                fitness = fitness.at[i].set(jnp.where(better, f_trial, fitness[i]))

                return pop, fitness, key

            pop, fitness, key = jax.lax.fori_loop(0, popsize, step, (pop, fitness, key))
            return pop, fitness, key

    elif updating == "deferred":

        def evolve(
            pop: jax.Array, fitness: jax.Array, key: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            keys = jax.random.split(key, popsize)
            trials, keys = jax.vmap(lambda i, k: make_trial(pop, i, k))(
                jnp.arange(popsize), keys
            )
            key = keys[-1]
            f_trials = pmapped_func(trials) if workers > 1 else jax.vmap(func)(trials)
            better = f_trials < fitness
            pop = jnp.where(better[:, None], trials, pop)
            fitness = jnp.where(better, f_trials, fitness)
            return pop, fitness, key

    else:
        msg = "updating must be 'immediate' or 'deferred'"
        raise ValueError(msg)

    def converged(fitness: jax.Array) -> jax.Array:
        return jnp.all(jnp.isfinite(fitness)) & (
            jnp.std(fitness) <= atol + tol * jnp.abs(jnp.mean(fitness))
        )

    nit, pop, fitness, key = jax.lax.while_loop(
        lambda val: (val[0] <= maxiter) & (~converged(val[2])),
        lambda val: (val[0] + 1, *evolve(val[1], val[2], val[3])),
        (1, pop, fitness, key),
    )

    best_idx = jnp.argmin(fitness)
    success = converged(fitness)
    return OptimizeResults(
        x=pop[best_idx],
        fun=fitness[best_idx],
        success=success,
        status=(~success).astype(int),
        jac=jnp.array(0),
        hess_inv=None,
        nfev=nit * popsize,
        njev=jnp.array(0),
        nit=nit,
    )
