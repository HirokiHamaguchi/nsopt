import jax.numpy as jnp
from .base import OptimizationProblem
from nsopt.regularizer.nuclear import NuclearNormRegularizer


def make_matrix_completion_problem(
    M: jnp.ndarray, Omega: jnp.ndarray, lam: float, X0: jnp.ndarray
) -> OptimizationProblem:
    """
    行列補完問題を作成する。
    M: 完成すべき行列
    Omega: 観測マスク
    lam: 正則化パラメータ
    X0: 初期推定値
    """

    # 行列サイズの取得
    n, m = M.shape

    # 目的関数 f(x)
    def f(x):
        X = jnp.reshape(x, (n, m))
        # 観測されたエントリのFrobeniusノルムの二乗誤差
        diff = Omega * (X - M)
        return 0.5 * jnp.sum(diff**2)

    return OptimizationProblem(f=f, regularizer=NuclearNormRegularizer(lam))
