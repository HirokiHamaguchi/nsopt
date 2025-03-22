import functools
import jax
import jax.numpy as jnp
from .base import ProblemBase

jax.config.update("jax_enable_x64", True)


class Problem(ProblemBase):
    def __init__(self, A, b, lam, x0):
        super().__init__()
        self.A = A  # Feature matrix
        self.b = b  # Labels (+1 or -1)
        self.lam = lam  # Regularization parameter
        self.x0 = x0  # Initial point

    @functools.partial(jax.jit, static_argnums=(0,))
    def f(self, x):
        # Logistic loss: sum(log(1 + exp(-b_i * a_i^T x)))
        # print(self.A.shape, x.shape) (100,10), (1,10)
        z = jnp.matmul(self.A, x.T).flatten()
        return jnp.mean(jnp.log(1 + jnp.exp(-self.b * z)))

    @functools.partial(jax.jit, static_argnums=(0,))
    def g(self, x):
        # L1 regularization term
        return self.lam * jnp.linalg.norm(x, 1)

    #   prox_{\eta g}(x)
    # = prox_{\eta \lambda ||\cdot||_1}(x)
    # = soft-thresholding(x, \eta \lambda)
    @functools.partial(jax.jit, static_argnums=(0,))
    def g_prox(self, x, eta):
        # Soft-thresholding for L1 norm
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - eta * self.lam, 0)
