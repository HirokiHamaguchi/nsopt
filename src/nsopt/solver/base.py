from nsopt.problem.base import OptimizationProblem

from .lbfgs import solveLBFGS


def solve(prob: OptimizationProblem, method: str, **kwargs):
    if method == "L-BFGS":
        solveLBFGS(prob, **kwargs)
    else:
        raise ValueError
