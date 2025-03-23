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


def randsrc(m, n, alphabet=None, prob=None):
    """
    Generate random matrix using prescribed alphabet.
    If alphabet and prob are None, generates a random bipolar matrix with values -1 or 1.
    """
    if alphabet is None:
        alphabet = [-1, 1]
    if prob is None:
        prob = [0.5, 0.5]
    return np.random.choice(alphabet, size=(m, n), p=prob)


def getdata(n, i, randstate):
    np.random.seed(randstate)

    m = n // 8

    # Generate xtrue
    k = n // 40
    xtrue = np.zeros(n)
    ind = np.random.choice(n, k, replace=False)
    eta1 = randsrc(k, 1).flatten()
    eta2 = np.random.rand(k)
    xtrue[ind] = eta1 * 10 ** (i * eta2)

    # Generate b = A*xtrue + noise, where A*xtrue = dct(xtrue)
    J = np.random.choice(n, m, replace=False)
    temp = dct(xtrue, norm="ortho")
    Axtrue = temp[J]
    noise = t.rvs(df=4, size=m)
    b = Axtrue + 1e-1 * noise

    # Calculate the dct matrix
    A = None
    if n <= 28500:
        B = np.eye(n)
        for j in range(n):
            B[j, :] = dct(np.eye(1, n, j).flatten(), norm="ortho")
        A = B[J, :]

    return xtrue, b, J, k, noise, A


# Parameters
n = 262144
tol = 1e-5
nu = 0.25
type = 1  # type=d/20 and d∈{20,40,60,80}, so you can choose it in {1,2,3,4}, which represents the dynamic range of signal
c_lam = 0.1  # choose c_lambda in {0.1,0.01}
nrun = 10

iter_array1 = np.zeros(nrun)
time_array1 = np.zeros(nrun)
fval_array1 = np.zeros(nrun)
resi_array1 = np.zeros(nrun)

for j in range(nrun):
    # **************** to fix the random seed **********************
    randstate = j * 100
    np.random.seed(randstate)

    # **************** get data **********************
    xtrue, b, J, k, noise, A = getdata(n, type, randstate)
    data = {"b": b, "J": J}
    m = len(b)
    normb = np.linalg.norm(b)
    Amap = lambda x: pdct(x, 1, n, J)
    ATmap = lambda x: pdct(x, 2, n, J)
    data["Amap"] = Amap
    data["ATmap"] = ATmap

    t = b / (nu + b**2)
    ATt = pdct(t, 2, n, J)
    lambda_ = c_lam * 2 * np.max(np.abs(ATt))

    model = {"loss": "student", "reg": "ell1", "nu": nu, "lambda": lambda_}

    # Generate an initial point
    x0 = pdct(b, 2, n, J)  # x0 = ATb

    # **************** parameter setting of IRPNM **********************
    solver = "DALMQ"
    OPTIONS = {
        "solver": solver,
        "maxiter": 1000,
        "maxiter_in": 100,
        "tol": tol,
        "printyes": 1,
    }

    pars = {"eta": 0.9, "b1": 1.0, "varrho": 0.45, "tau": 0.45}

    # The IRPNM function is not implemented as per your instructions.
    # Assuming IRPNM is a placeholder for the actual optimization algorithm.
    # xopt, Fopt, resi, iter, ttime = IRPNM(x0, data, model, OPTIONS, pars)

    # For demonstration purposes, we'll assign dummy values to these variables.
    xopt = x0  # Placeholder
    Fopt = 0  # Placeholder
    resi = 0  # Placeholder
    iter = 1  # Placeholder
    ttime = 0  # Placeholder

    iter_array1[j] = iter - 1
    time_array1[j] = ttime
    fval_array1[j] = Fopt
    resi_array1[j] = resi

iter1 = np.mean(iter_array1)
time1 = np.mean(time_array1)
fval1 = np.mean(fval_array1)
resi1 = np.mean(resi_array1)

print("---------------------------------------------------------------------------")
print("   Algorithm  |  iter  |  CPU time  |     Obj.    |   Residual  |   Rate    ")
print("---------------------------------------------------------------------------")
print(f'{"IRPNM":12s}: \t {iter1:.1f} \t  {time1:.1f} \t  {fval1:.4f} \t  {resi1:.2e} ')
