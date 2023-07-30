import numpy as np


# Rosenbrock function and its gradient
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


def rosenbrock_gradient(x):
    return np.array([-400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]), 200 * (x[1] - x[0]**2)])


def hessian_approx(x, f):
    h = 10**(-8)
    b_k = np.empty([2, 2])

    e_1 = np.array([1, 0])
    e_2 = np.array([0, 1])

    b_k[0, 0] = (f(x + h * e_1)[0] - f(x)[0]) / h
    b_k[1, 0] = (f(x + h * e_2)[0] - f(x)[0]) / h
    b_k[0, 1] = (f(x + h * e_1)[1] - f(x)[1]) / h
    b_k[1, 1] = (f(x + h * e_2)[1] - f(x)[1]) / h

    return b_k


vec = np.array([1, 1])

print(hessian_approx(vec, rosenbrock_gradient))


