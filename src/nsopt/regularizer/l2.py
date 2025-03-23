import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Any


# L2正則化の例: g(x)=0.5 * lambda * ||x||^2, proxは単純な縮小作用
class L2Regularizer(Regularizer):
    def __init__(self, lam: float):
        self.lam = lam

    def g(self, x: Any) -> float:
        return 0.5 * self.lam * jnp.sum(x**2)

    def subgrad_g(self, x):
        return self.lam * x

    def prox(self, x: Any, alpha: float) -> Any:
        return x / (1.0 + alpha * self.lam)
