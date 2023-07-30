import numpy as np
from scipy.optimize import minimize


def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def rosenbrock_gradient(x):
    return np.array([-400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]), 200 * (x[1] - x[0]**2)])


def sphere(x):
    return x[0]**2 + x[1]**2


def sphere_gradient(x):
    return np.array([2 * x[0], 2 * x[1]])


def mccormick(x):
    return np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5 * x[0] + 2.5 * x[1] + 1


def mccormick_gradient(x):
    return np.array([np.cos(x[0] + x[1]) + 2 * (x[0] - x[1]) - 1.5,
                     np.cos(x[0] + x[1]) - 2 * (x[0] - x[1]) + 2.5])


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


def m_k(fx, grad_fx, B, p):
    return fx + np.dot(grad_fx, p) + 0.5 * np.dot(p, np.dot(B, p))


def rtrm(f, grad_f, x_0, delta_0=0.5, eta_1=0.15, eta_tilde_1=0.9,
                               eta_tilde_2=0.9, gamma_1=0.9, gamma_2=0.9, tol=1e-6):
    x_k = x_0
    delta_k = delta_0
    k = 0
    x_prev = None
    delta_prev = None

    while np.linalg.norm(grad_f(x_k)) > tol:
        # Step 1: Model definition
        fx_k = f(x_k)
        grad_fx_k = grad_f(x_k)
        B_k = hessian_approx(x_k, grad_f)

        # Step 2: Retrospective trust-region radius update
        if k > 0:
            if np.all(x_k == x_prev):
                # delta_k = np.random.uniform(gamma_1 * delta_prev, gamma_2 * delta_prev)
                delta_k = gamma_1 * delta_prev
            else:
                rho_tilde_k = (f(x_prev) - fx_k) / \
                              (m_k(f(x_prev), grad_f(x_prev), B_k, x_prev - x_k) - m_k(fx_k, grad_fx_k, B_k, x_prev - x_k))
                if rho_tilde_k >= eta_tilde_2:
                    delta_k = delta_prev
                elif eta_tilde_1 <= rho_tilde_k < eta_tilde_2:
                    delta_k = gamma_2 * delta_prev
                else:
                    delta_k = gamma_1 * delta_prev

        # Step 3: Step calculation
        s_k = -delta_k * grad_fx_k / np.linalg.norm(grad_fx_k)

        # Step 4: Acceptance of the trial point
        rho_k = (fx_k - f(x_k + s_k)) / \
                (m_k(fx_k, grad_fx_k, B_k, np.zeros_like(x_k)) - m_k(fx_k, grad_fx_k, B_k, s_k))
        if rho_k > eta_1:
            x_next = x_k + s_k
        else:
            x_next = x_k

        x_prev = x_k
        x_k = x_next
        delta_prev = delta_k
        k += 1

    return x_k, k


x0 = np.array([3.0, -2.9])
result_b, num_b = rtrm(rosenbrock, rosenbrock_gradient, x0)
result_s, num_s = rtrm(sphere, sphere_gradient, x0)
result_m, num_m = rtrm(mccormick, mccormick_gradient, x0)
print("Optimal point for Rosenbrock:", result_b)
print("Number of iterations for Rosenbrock:", num_b)
print("Optimal point for Sphere:", result_s)
print("Number of iterations for Sphere:", num_s)
print("Optimal point for McCormick:", result_m)
print("Number of iterations for McCormick:", num_m)
