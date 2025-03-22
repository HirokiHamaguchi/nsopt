import functools
import jax
import jax.numpy as jnp
from .base import ProblemBase

jax.config.update("jax_enable_x64", True)


class Problem(ProblemBase):
    def __init__(self, M, Omega, lam, X0):
        super().__init__()
        self.M = M
        self.Omega = Omega  # Observation mask
        self.lam = lam  # Regularization parameter
        self.x0 = X0.flatten()

    @functools.partial(jax.jit, static_argnums=(0,))
    def f(self, x):
        X = jnp.reshape(x, self.M.shape)
        # Squared Frobenius norm loss on observed entries
        diff = self.Omega * (X - self.M)
        return 0.5 * jnp.sum(diff**2)

    @functools.partial(jax.jit, static_argnums=(0,))
    def g(self, x):
        X = jnp.reshape(x, self.M.shape)
        # Nuclear norm regularization term
        return self.lam * jnp.sum(jnp.linalg.svd(X, compute_uv=False))

    @functools.partial(jax.jit, static_argnums=(0,))
    def g_prox(self, x, eta):
        X = jnp.reshape(x, self.M.shape)
        # Singular Value Thresholding (SVT): prox for nuclear norm
        U, S, Vt = jnp.linalg.svd(X, full_matrices=False)
        # Soft-thresholding on singular values
        S_thresh = jnp.maximum(S - eta * self.lam, 0)
        ret = U @ jnp.diag(S_thresh) @ Vt
        return ret.flatten()
