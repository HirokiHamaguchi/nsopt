import functools
import jax
import jax.numpy as jnp
from .base import OptimizationProblem
from nsopt.regularizer.l1 import L1Regularizer


# 例1: LASSO問題
# f(x)=0.5*||Ax - b||^2, g(x)=lambda * ||x||_1  (L1正則化)
def make_lasso_problem(
    A: jnp.ndarray, b: jnp.ndarray, lam: float
) -> OptimizationProblem:
    def f(x):
        # Aが行列、xがベクトルとする。np.dotでも可。
        coeff = 1.0 / (2 * A.shape[0])
        return coeff * jnp.linalg.norm(jnp.dot(A, x) - b) ** 2

    regularizer = L1Regularizer(lam)
    return OptimizationProblem(f=f, regularizer=regularizer)
