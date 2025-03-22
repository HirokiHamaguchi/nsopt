import numpy as np


class ProximalFunction:
    """Base class for functions with proximal operators"""

    def func(self, x):
        """Evaluate function f(x)"""
        raise NotImplementedError

    def grad(self, x):
        """Evaluate gradient ∇f(x)"""
        raise NotImplementedError

    def prox(self, x, lambda_):
        """Evaluate proximal operator prox_{lambda f}(x)"""
        raise NotImplementedError


class Zero(ProximalFunction):
    """0"""

    def __init__(self):
        pass

    def func(self, x):
        return 0

    def grad(self, x):
        return 0

    def prox(self, x):
        return 0


class L1Norm(ProximalFunction):
    """L1 Norm: f(x) = λ ||x||_1"""

    def __init__(self, lam=1.0):
        self.lam = lam

    def func(self, x):
        return self.lam * np.sum(np.abs(x))

    def grad(self, x):
        """Subgradient of L1 norm"""
        return self.lam * np.sign(x)

    def prox(self, x, lambda_):
        """Soft-thresholding operation"""
        return np.sign(x) * np.maximum(np.abs(x) - lambda_ * self.lam, 0)


class L2NormSquared(ProximalFunction):
    """L2 Norm Squared: f(x) = (1/2) ||x||_2^2"""

    def func(self, x):
        return 0.5 * np.sum(x**2)

    def grad(self, x):
        return x  # Gradient of (1/2) ||x||_2^2 is x

    def prox(self, x, lambda_):
        return x / (1 + lambda_)  # Simple scaling


class L2Norm(ProximalFunction):
    """L2 Norm: f(x) = λ ||x||_2"""

    def __init__(self, lam=1.0):
        self.lam = lam

    def func(self, x):
        return self.lam * np.linalg.norm(x, 2)

    def grad(self, x):
        norm_x = np.linalg.norm(x, 2)
        if norm_x == 0:
            return np.zeros_like(x)
        return self.lam * x / norm_x

    def prox(self, x, lambda_):
        norm_x = np.linalg.norm(x, 2)
        if norm_x == 0:
            return x
        return max(norm_x - lambda_ * self.lam, 0) * x / norm_x


class L0Norm(ProximalFunction):
    """L0 Norm: f(x) = λ ||x||_0"""

    def __init__(self, lam=1.0):
        self.lam = lam

    def func(self, x):
        return self.lam * np.count_nonzero(x)

    def grad(self, x):
        """L0 norm has no well-defined gradient"""
        raise NotImplementedError("L0 norm is non-differentiable everywhere.")

    def prox(self, x, lambda_):
        """Hard-thresholding"""
        return np.where(np.abs(x) > np.sqrt(2 * lambda_ * self.lam), x, 0)


class IndicatorNonNegative(ProximalFunction):
    """Indicator function of x >= 0: f(x) = 0 if x >= 0, else ∞"""

    def func(self, x):
        return 0 if np.all(x >= 0) else np.inf

    def grad(self, x):
        raise NotImplementedError("Indicator function is not differentiable.")

    def prox(self, x, lambda_):
        """Projection onto non-negative orthant"""
        return np.maximum(x, 0)


class NuclearNorm(ProximalFunction):
    """Nuclear Norm: f(X) = λ ||X||_*"""

    def __init__(self, lam=1.0):
        self.lam = lam

    def func(self, X):
        return self.lam * np.sum(np.linalg.svd(X, compute_uv=False))

    def grad(self, X):
        """Gradient of the nuclear norm is the subgradient"""
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        return self.lam * U @ Vt

    def prox(self, X, lambda_):
        """Singular value soft-thresholding"""
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        S_thresholded = np.maximum(S - lambda_ * self.lam, 0)
        return U @ np.diag(S_thresholded) @ Vt
