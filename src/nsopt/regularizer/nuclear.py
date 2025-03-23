import jax.numpy as jnp
import jax
from nsopt.regularizer.base import Regularizer
from nsopt.problem.base import OptimizationProblem


# 核ノルム (Nuclear Norm) 正則化
class NuclearNormRegularizer(Regularizer):
    def __init__(self, lam: float):
        self.lam = lam

    def g(self, x):
        """核ノルム (特異値の和)"""
        return self.lam * jnp.sum(jnp.linalg.svd(x, compute_uv=False))

    def subgrad_g(self, X):
        U, S, Vh = jnp.linalg.svd(X)
        S_inv = 1.0 / (S + 1e-8)  # 1e-8でゼロ除算を防ぐ
        subgrad = U @ jnp.diag(S_inv) @ Vh
        return self.lam * subgrad

    def prox(self, x, alpha):
        """特異値ソフトしきい値処理 (Singular Value Thresholding, SVT)"""
        U, S, Vt = jnp.linalg.svd(x, full_matrices=False)
        S_thresholded = jnp.maximum(S - alpha * self.lam, 0.0)
        return U @ jnp.diag(S_thresholded) @ Vt
