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


# initial points
x0 = np.array([3.0, -2.9])
delta_initial = 0.5
eta = 0.15
eta_tilde_1 = 0.25
eta_tilde_2 = 0.35
gamma_1 = 0.2
gamma_2 = 0.4


def rtrm(f, grad_f, x_0, delta, tol=1e-6):
    x = np.array(x0)
    f_x0 = f(x)
    k = 0

    def m_k(fx, grad_fx, B, p):
        return fx + np.dot(grad_fx, p) + 0.5 * np.dot(p, np.dot(B, p))

    def cauchy_point(g, B, delta):
        p = - delta / np.linalg.norm(g) * g
        if np.dot(g, np.dot(B, g)) <= 0:
            tau = 1
        else:
            tau = min(1, np.linalg.norm(g)**3 / (delta * np.dot(g, np.dot(B, g))))
        return tau * p

    if k == 0:
        s_k = cauchy_point(grad_f(x), hessian_approx(x, f), delta)
        x_new = x + s_k








