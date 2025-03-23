import jax
import jax.numpy as jnp
from typing import Callable, Any
from nsopt.regularizer.base import Regularizer
from nsopt.solver.base import solve


class OptimizationProblem:
    def __init__(
        self,
        f: Callable[[Any], float],
        regularizer: Regularizer,
        x0: Any,
        name: str,
    ):
        self.f = f
        self.f_grad = jax.grad(self.f)
        self.regularizer = regularizer
        self.x0 = x0
        self.name = name

    def __str__(self):
        return f"OptimizationProblem({self.name})"

    def __repr__(self):
        return f"OptimizationProblem({self.name})"

    def obj(self, x: Any) -> float:
        return self.f(x) + self.g(x)

    def subgrad_obj(self, x: Any) -> float:
        return self.grad_f(x) + self.subgrad_g(x)

    def f(self, x: Any) -> float:
        """目的関数 f(x) + g(x) を評価する。xはnp.arrayまたはjnp.arrayを受け付ける"""
        return self.f(x)

    def grad_f(self, x: Any) -> Any:
        """fの勾配を計算する"""
        return self.f_grad(x)

    def g(self, x: Any) -> float:

        return self.regularizer.g(x)

    def subgrad_g(self, x: Any) -> float:

        return self.regularizer.subgrad_g(x)

    def prox_g(self, x: Any, alpha: float) -> Any:
        """正則化項のproximal operatorを計算する"""

        return self.regularizer.prox(x, alpha)

    def solve(self, method, **kwargs):
        """Compiles and solves the problem using the specified method."""
        return solve(self, method, **kwargs)
