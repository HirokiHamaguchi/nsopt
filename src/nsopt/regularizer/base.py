import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Any


# 正則化項のインターフェース（抽象クラス）: g(x) とprox_g(x, alpha)を定義
class Regularizer:
    def g(self, x: Any) -> float:
        """正則化項の評価値を返す"""
        raise NotImplementedError

    def subgrad_g(self, x: Any) -> float:
        return NotImplementedError

    def prox(self, x: Any, alpha: float) -> Any:
        """proximal operatorを計算する"""
        raise NotImplementedError
