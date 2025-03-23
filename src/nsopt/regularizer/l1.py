import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Any
from .base import Regularizer


# L1正則化の例: g(x)=lambda * ||x||_1, proxはsoft thresholding
class L1Regularizer(Regularizer):
    def __init__(self, lam: float):
        self.lam = lam

    def g(self, x: Any) -> float:
        return self.lam * jnp.sum(jnp.abs(x))

    def subgrad_g(self, x):
        return self.lam * jnp.sign(x)

    #   prox_{\eta g}(x)
    # = prox_{\eta \lambda ||\cdot||_1}(x)
    # = soft-thresholding(x, \eta \lambda)
    def prox(self, x: Any, eta: float) -> Any:

        threshold = eta * self.lam
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0.0)
