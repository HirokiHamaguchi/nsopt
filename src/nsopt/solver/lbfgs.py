from nsopt.problem.base import OptimizationProblem
import scipy.optimize


def solveLBFGS(prob: OptimizationProblem):
    return scipy.optimize.minimize(
        prob.obj, prob.x0, method="L-BFGS", jac=prob.obj_subgrad
    )
