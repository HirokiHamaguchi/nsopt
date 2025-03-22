import functools
import jax


class ProblemBase:
    def __init__(self):
        pass

    @functools.partial(jax.jit, static_argnums=(0,))
    def f(self, x):
        raise NotImplementedError

    @functools.partial(jax.jit, static_argnums=(0,))
    def g(self, x):
        raise NotImplementedError

    @functools.partial(jax.jit, static_argnums=(0,))
    def g_prox(self, x, eta):
        raise NotImplementedError
