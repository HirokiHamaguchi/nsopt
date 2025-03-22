import functools
import jax
import jax.numpy as jnp
from .base import ProblemBase

jax.config.update("jax_enable_x64", True)


class Problem(ProblemBase):
    def __init__(self, A, b, lam, x0):
        super().__init__()
        self.A = A
        self.b = b
        self.lam = lam
        self.x0 = x0

    @functools.partial(jax.jit, static_argnums=(0,))
    def f(self, x):
        coeff = 1.0 / (2 * self.A.shape[0])
        return coeff * jnp.linalg.norm(jnp.dot(self.A, x) - self.b) ** 2

    @functools.partial(jax.jit, static_argnums=(0,))
    def g(self, x):
        return self.lam * jnp.linalg.norm(x, 1)

    #   prox_{\eta g}(x)
    # = prox_{\eta \lambda ||\cdot||_1}(x)
    # = soft-thresholding(x, \eta \lambda)
    @functools.partial(jax.jit, static_argnums=(0,))
    def g_prox(self, x, eta):
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - eta * self.lam, 0)
