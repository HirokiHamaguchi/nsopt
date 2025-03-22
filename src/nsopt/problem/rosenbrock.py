import jax
import jax.numpy as jnp
import functools
from .base import ProblemBase

jax.config.update("jax_enable_x64", True)


class Problem(ProblemBase):
    def __init__(self, d, x0=0.5, a=1.0, b=100.0):
        super().__init__()
        self.d = d
        self.x0 = x0 * jnp.ones(d)
        self.a = a
        self.b = b

    @functools.partial(jax.jit, static_argnums=(0,))
    def f(self, x):
        # \sum_{i=1}^{d-1} (a - x_i)^2 + b(x_{i+1} - x_i^2)^2
        r = jnp.concatenate((self.a - x[:-1], jnp.sqrt(self.b) * (x[1:] - x[:-1] ** 2)))
        return jnp.sum(jnp.square(r))

    def f_and_f_grad(self, x):
        return self.f(x), jax.grad(self.f)(x)

    @functools.partial(jax.jit, static_argnums=(0,))
    def g(self, x):
        # Special case: g(x) = 0
        return 0

    @functools.partial(jax.jit, static_argnums=(0,))
    def g_prox(self, x, eta):
        return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # hessianを描画する imshowで
    d = 10
    problem = Problem(d)
    x = problem.x0
    H = jax.hessian(problem.f)(x)
    plt.imshow(H)
    plt.colorbar()
    plt.show()
