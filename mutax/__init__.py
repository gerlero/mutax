"""Evolutionary optimization algorithms in JAX."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.optimize

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
def differential_evolution(  # noqa: PLR0913
    func: Callable[[jax.Array], jax.Array],
    /,
    bounds: jax.Array,
    *,
    key: jax.Array,
    maxiter: int = 1_000,
    popsize: int = 15,
    mutation: float | tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.8,
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
    - `mutation`: A float or a tuple of two floats specifying the mutation factor. If a
    tuple is provided, the mutation factor is sampled uniformly from this range for each
    mutation.
    - `recombination`: A float in [0, 1] specifying the recombination probability.

    **Returns:**
    An `OptimizeResults` object containing the optimization results.
    """
    dim = len(bounds)
    lower = jnp.array([b[0] for b in bounds])
    upper = jnp.array([b[1] for b in bounds])
    popsize *= dim

    # Initialize population
    key, subkey = jax.random.split(key)
    pop = jax.random.uniform(subkey, (popsize, dim), minval=lower, maxval=upper)
    fitness: jax.Array = jax.vmap(func)(pop)

    def evolve(
        pop: jax.Array, fitness: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        def step(
            i: int, carry: tuple[jax.Array, jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            pop, fitness, key = carry
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
                mut_val = jax.random.uniform(
                    subkey, (), minval=mut_lower, maxval=mut_upper
                )

            mutant = pop[r1] + mut_val * (pop[r2] - pop[r3])
            mutant = jnp.clip(mutant, lower, upper)

            # Crossover
            key, subkey = jax.random.split(key)
            cross_points = jax.random.uniform(subkey, (dim,)) < recombination
            key, subkey = jax.random.split(key)
            cross_points = cross_points.at[jax.random.randint(subkey, (), 0, dim)].set(
                True
            )
            trial = jnp.where(cross_points, mutant, pop[i])

            # Selection
            f_trial = func(trial)
            better = f_trial < fitness[i]
            pop = pop.at[i].set(jnp.where(better, trial, pop[i]))
            fitness = fitness.at[i].set(jnp.where(better, f_trial, fitness[i]))

            return pop, fitness, key

        pop, fitness, key = jax.lax.fori_loop(0, popsize, step, (pop, fitness, key))
        return pop, fitness, key

    pop, fitness, key = jax.lax.fori_loop(
        0, maxiter, lambda _, val: evolve(*val), (pop, fitness, key)
    )

    best_idx = jnp.argmin(fitness)
    return OptimizeResults(
        x=pop[best_idx],
        fun=fitness[best_idx],
        success=True,
        status=0,
        jac=jnp.array(0),
        hess_inv=None,
        nfev=maxiter * popsize,
        njev=jnp.array(0),
        nit=maxiter,
    )
