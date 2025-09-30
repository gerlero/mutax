"""Evolutionary optimization algorithms in JAX."""

import warnings
from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.optimize
from parajax import autopmap

OptimizeResults = jax.scipy.optimize.OptimizeResults
"""Object holding optimization results.

**Attributes:**

- `x`: final solution.
- `success`: whether the optimization succeeded.
- `status`: integer solver specific return code. 0 means converged (nominal),
  1=max number of iterations reached.
- `fun`: final function value.
- `nfev`: integer number of function calls used.
- `njev`: integer number of Jacobian evaluations used (only if `polish` was
  set to ``True``).
- `nit`: integer number of iterations of the optimization algorithm.
- `jac`: final Jacobian (only if the solution was polished).
- `hess_inv`: inverse of the final Hessian (only if the solution was polished).
"""


@eqx.filter_jit
def differential_evolution(  # noqa: C901, PLR0913, PLR0915
    func: Callable[[jax.Array], jax.Array],
    /,
    bounds: jax.Array,
    *,
    key: jax.Array | None = None,
    strategy: Literal["rand1bin", "best1bin"] = "best1bin",
    maxiter: int = 1_000,
    popsize: int = 15,
    tol: float = 0.01,
    atol: float = 0,
    mutation: float | tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.8,
    disp: bool = False,
    polish: bool = True,
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
    - `key`: A JAX random key for stochastic operations. You can use e.g.
      `jax.random.key(seed)` to generate a key. If not given, a default key is used.
    - `strategy`: The differential evolution strategy to use. Can be either "rand1bin"
      or "best1bin". The "rand1bin" strategy uses a randomly selected population member
      as the base vector, while "best1bin" uses the best population member found so far.
    - `maxiter`: The maximum number of generations to evolve the population.
    - `popsize`: Multiplier for setting the total population size. The population size
      is determined by `popsize * dim`.
    - `tol`: Relative tolerance for convergence.
    - `atol`: Absolute tolerance for convergence.
    - `mutation`: A float or a tuple of two floats specifying the mutation factor. If a
      tuple is provided, the mutation factor is sampled uniformly from this range for
      each mutation.
    - `recombination`: A float in [0, 1] specifying the recombination probability.
    - `disp`: Whether to print progress messages at each iteration.
    - `polish`: Whether to perform a local optimization using BFGS at the end of the
      evolution process to attempt to refine the best solution found. For this local
      optimization to be effective, the objective function should be differentiable.
    - `updating`: Strategy for updating the population. Can be either "immediate" or
      "deferred". "immediate" updates individuals as soon as a better trial vector is
      found, while "deferred" updates the population after all trial vectors have been
      evaluated.
    - `workers`: Number of JAX devices (CPUs/GPUs/TPUs) used for evaluating the
      objective function. If set to -1, uses all available JAX devices. Uses
      [Parajax](https://github.com/gerlero/parajax) for parallelization. Note that
      setting this to a value other than 1 will override `updating` to "deferred".
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

    if key is None:
        key = jax.random.key(8959915698270734364)  # = hash("mutax")

    if workers < 1 and workers != -1:
        msg = "workers must be a positive integer or -1"
        raise ValueError(msg)

    if workers != 1 and updating == "immediate":
        msg = (
            "differential_evolution: the 'workers' keyword has overridden "
            "updating='immediate' to updating='deferred'"
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        updating = "deferred"

    vmapped_func = jax.vmap(func)
    if workers != 1:
        vmapped_func = autopmap(
            vmapped_func, max_devices=None if workers == -1 else workers
        )

    # Initialize population (Latin hypercube sampling)
    segsize = 1.0 / popsize
    key, subkey = jax.random.split(key)
    pop = lower + (upper - lower) * jnp.stack(
        jax.vmap(
            lambda k: (
                segsize * jax.random.uniform(k, (popsize,))
                + jnp.linspace(0.0, 1.0, popsize, endpoint=False)
            )[jax.random.permutation(k, popsize)]
        )(jax.random.split(subkey, dim)),
        axis=1,
    )

    if x0 is not None:
        pop = pop.at[0].set(jnp.asarray(x0))

    fitness = vmapped_func(pop)

    def make_trial(
        pop: jax.Array, fitness: jax.Array, i: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        key, subkey = jax.random.split(key)

        if strategy == "best1bin":
            # Use best member as base vector
            best_idx = jnp.argmin(fitness)

            # Select two distinct indices from 0..pop_size-1 excluding i and best_idx
            idxs = jnp.arange(popsize)
            idxs = jnp.where(idxs == i, popsize, idxs)
            idxs = jnp.where(idxs == best_idx, popsize + 1, idxs)
            idx_perm = jax.random.permutation(subkey, idxs)
            r1, r2 = idx_perm[:2]
            r1 = jnp.where(r1 == popsize, idx_perm[2], r1)
            r1 = jnp.where(r1 == popsize + 1, idx_perm[3], r1)
            r2 = jnp.where(r2 == popsize, idx_perm[4], r2)
            r2 = jnp.where(r2 == popsize + 1, idx_perm[5], r2)

            # Mutation
            try:
                mut_lower, mut_upper = mutation  # ty: ignore[not-iterable]
            except TypeError:
                mut_val = mutation
            else:
                key, subkey = jax.random.split(key)
                mut_val = jax.random.uniform(
                    subkey, (), minval=mut_lower, maxval=mut_upper
                )

            mutant = pop[best_idx] + mut_val * (pop[r1] - pop[r2])

        elif strategy == "rand1bin":
            # Use random member as base vector
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
                mut_val = jax.random.uniform(
                    subkey, (), minval=mut_lower, maxval=mut_upper
                )

            mutant = pop[r1] + mut_val * (pop[r2] - pop[r3])

        else:
            msg = f"Unrecognized strategy '{strategy}'"
            raise ValueError(msg)

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
            nit: int, pop: jax.Array, fitness: jax.Array, key: jax.Array
        ) -> tuple[int, jax.Array, jax.Array, jax.Array]:
            if disp:
                jax.debug.print(
                    "differential_evolution step {nit}: f(x)={fmin}",
                    nit=nit,
                    fmin=jnp.min(fitness),
                )

            def evolve_one(
                i: int, carry: tuple[jax.Array, jax.Array, jax.Array]
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                pop, fitness, key = carry
                trial, key = make_trial(pop, fitness, i, key)

                # Selection
                f_trial = func(trial)
                better = f_trial < fitness[i]
                pop = pop.at[i].set(jnp.where(better, trial, pop[i]))
                fitness = fitness.at[i].set(jnp.where(better, f_trial, fitness[i]))

                return pop, fitness, key

            pop, fitness, key = jax.lax.fori_loop(
                0, popsize, evolve_one, (pop, fitness, key)
            )
            return nit + 1, pop, fitness, key

    elif updating == "deferred":

        def evolve(
            nit: int, pop: jax.Array, fitness: jax.Array, key: jax.Array
        ) -> tuple[int, jax.Array, jax.Array, jax.Array]:
            if disp:
                jax.debug.print(
                    "differential_evolution step {nit}: f(x)={fmin}",
                    nit=nit,
                    fmin=jnp.min(fitness),
                )

            keys = jax.random.split(key, popsize)
            trials, keys = jax.vmap(lambda i, k: make_trial(pop, fitness, i, k))(
                jnp.arange(popsize), keys
            )
            key = keys[-1]
            f_trials = vmapped_func(trials)
            better = f_trials < fitness
            pop = jnp.where(better[:, None], trials, pop)
            fitness = jnp.where(better, f_trials, fitness)
            return nit + 1, pop, fitness, key

    else:
        msg = "updating must be 'immediate' or 'deferred'"
        raise ValueError(msg)

    def converged(fitness: jax.Array) -> jax.Array:
        return jnp.all(jnp.isfinite(fitness)) & (
            jnp.std(fitness) <= atol + tol * jnp.abs(jnp.mean(fitness))
        )

    nit, pop, fitness, key = jax.lax.while_loop(
        lambda val: (val[0] <= maxiter) & (~converged(val[2])),
        lambda val: evolve(*val),
        (1, pop, fitness, key),
    )

    success = converged(fitness)
    best_idx = jnp.argmin(fitness)
    best = pop[best_idx]
    best_fitness = fitness[best_idx]

    if polish:
        if disp:
            jax.debug.print("Polishing solution with BFGS")

        result = jax.scipy.optimize.minimize(
            func,
            best,
            method="BFGS",
        )
        polished = (
            result.success
            & (result.fun < best_fitness)
            & jnp.all((result.x >= lower) & (result.x <= upper))
        )
        best = jnp.where(polished, result.x, best)
        best_fitness = jnp.where(polished, result.fun, best_fitness)

        return OptimizeResults(
            x=best,
            fun=best_fitness,
            success=success,
            status=(~success).astype(int),
            jac=jnp.where(polished, result.jac, jnp.full_like(result.jac, jnp.nan)),
            hess_inv=jnp.where(
                polished, result.hess_inv, jnp.full_like(result.hess_inv, jnp.nan)
            ),
            nfev=result.nfev + nit * popsize,
            njev=result.njev,
            nit=nit,
        )

    return OptimizeResults(
        x=best,
        fun=best_fitness,
        success=success,
        status=(~success).astype(int),
        jac=None,
        hess_inv=None,
        nfev=nit * popsize,
        njev=jnp.array(0),
        nit=nit,
    )
