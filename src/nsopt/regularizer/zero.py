import jax.numpy as jnp
from typing import Any
from .base import Regularizer


class ZeroRegularizer(Regularizer):
    def __init__(self):
        pass

    def g(self, x: Any) -> float:
        return 0

    def subgrad_g(self, x):
        return jnp.zeros_like(x)

    def prox(self, x: Any) -> Any:
        return x
