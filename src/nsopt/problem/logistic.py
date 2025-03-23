import jax.numpy as jnp
from .base import OptimizationProblem
from nsopt.regularizer.l2 import L2Regularizer


# 例2: logistic regression問題
# f(x)=Σ(log(1+exp(Ax)) - b*(Ax)), g(x)=0.5*lambda*||x||^2  (L2正則化)
def make_logistic_regression_problem(
    A: jnp.ndarray, b: jnp.ndarray, lam: float
) -> OptimizationProblem:
    # self.A = A  # Feature matrix
    # self.b = b  # Labels (+1 or -1)
    # self.lam = lam  # Regularization parameter
    # self.x0 = x0  # Initial point

    def f(x):
        logits = jnp.dot(A, x)
        return jnp.sum(jnp.log(1 + jnp.exp(logits)) - b * logits)

    regularizer = L2Regularizer(lam)
    return OptimizationProblem(f=f, regularizer=regularizer)
