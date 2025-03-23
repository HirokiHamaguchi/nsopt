import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Any
from .base import Regularizer


# Elastic Net 正則化 (L1 + L2)
class ElasticNetRegularizer(Regularizer):
    def __init__(self, l1_ratio: float, lam: float):
        self.l1_ratio = l1_ratio
        self.lam = lam
        self.l1 = l1_ratio * lam
        self.l2 = (1 - l1_ratio) * lam

    def g(self, x):
        return self.l1 * jnp.sum(jnp.abs(x)) + 0.5 * self.l2 * jnp.sum(x**2)

    def subgrad_g(self, x):
        subgrad_l1 = self.lam1 * jnp.sign(x)
        subgrad_l2 = self.lam2 * x
        return subgrad_l1 + subgrad_l2

    def prox(self, x, alpha):
        threshold = alpha * self.l1
        x_shrink = jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0.0)
        return x_shrink / (1 + alpha * self.l2)
