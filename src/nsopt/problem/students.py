import numpy as np
from scipy.fft import dct, idct
from scipy.stats import t


def pdct(x, mode, n, picks):
    """
    Implements partial cosine transform matrix defined by taking only the
    rows in the picks vector. Resulting matrix A is m x n, m < n.
    pdct returns the following quantities based on the value of mode:

    -1: y = picks
     1: y = A*x
     2: y = A'*x
    """
    if mode == -1:
        return picks
    elif mode == 1:
        # y = A*x
        y = dct(x, norm="ortho")
        return y[picks]
    elif mode == 2:
        # y = A'*x
        y = np.zeros(n)
        y[picks] = x
        return idct(y, norm="ortho")
    else:
        raise ValueError("mode must be 1 (for A*x) or 2 (for A'*x).")


# Parameters
n = 512**2
tol = 1e-5
nu = 0.25
type = 1  # type=d/20 and d∈{20,40,60,80}, so you can choose it in {1,2,3,4}, which represents the dynamic range of signal
c_lam = 0.1  # choose c_lambda in {0.1, 0.01}
seed = 0
np.random.seed(seed)

m = n // 8

# Generate xtrue
k = n // 40
xtrue = np.zeros(n)
ind = np.random.choice(n, k, replace=False)
eta1 = np.random.choice([-1, 1], size=k)
eta2 = np.random.rand(k)
xtrue[ind] = eta1 * 10 ** (type * eta2)

# Generate b = A*xtrue + noise, where A*xtrue = dct(xtrue)
J = np.random.choice(n, m, replace=False)
temp = dct(xtrue, norm="ortho")
Axtrue = temp[J]
noise = t.rvs(df=4, size=m)
b = Axtrue + 1e-1 * noise

normb = np.linalg.norm(b)
t = b / (nu + b**2)
ATt = pdct(t, 2, n, J)
lambda_ = c_lam * 2 * np.max(np.abs(ATt))

model = {"loss": "student", "reg": "ell1", "nu": nu, "lambda": lambda_}

# Generate an initial point
x0 = pdct(b, 2, n, J)  # x0 = ATb
