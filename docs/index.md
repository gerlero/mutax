# Documentation for [`mutax`](https://github.com/gerlero/mutax)

!!! note "Enabling parallel execution on CPU"

    By default, JAX on CPU only uses a single core. To enable parallel execution on all available CPU cores, [set the `jax_num_cpu_devices` configuration option](https://docs.jax.dev/en/latest/sharded-computation.html) appropriately. This should be done at the beginning of your code as follows:

    ```python
    import multiprocessing
    import jax

    jax.config.update("jax_num_cpu_devices", multiprocessing.cpu_count())
    ```

### ::: mutax.differential_evolution


### ::: mutax.OptimizeResults
