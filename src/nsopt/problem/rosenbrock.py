import jax
import jax.numpy as jnp

from nsopt.problem.base import OptimizationProblem
from nsopt.regularizer.zero import ZeroRegularizer


def make_rosenbrock_problem(
    d: int,
    a: float = 1.0,
    b: float = 100.0,
):
    def f(x):
        r = jnp.concatenate((a - x[:-1], jnp.sqrt(b) * (x[1:] - x[:-1] ** 2)))
        return jnp.sum(jnp.square(r))

    regularizer = ZeroRegularizer()
    x0 = 0.5 * jnp.ones(d)
    return OptimizationProblem(f, regularizer, x0, "rosenbrock")
