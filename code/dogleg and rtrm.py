import numpy as np
import matplotlib.pyplot as plt


def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


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


def hessian_approx(x, f):  # from Homework 2
    h = 10**(-8)
    b_k = np.empty([2, 2])

    e_1 = np.array([1, 0])
    e_2 = np.array([0, 1])

    b_k[0, 0] = (f(x + h * e_1)[0] - f(x)[0]) / h
    b_k[1, 0] = (f(x + h * e_2)[0] - f(x)[0]) / h
    b_k[0, 1] = (f(x + h * e_1)[1] - f(x)[1]) / h
    b_k[1, 1] = (f(x + h * e_2)[1] - f(x)[1]) / h

    return b_k


def m_k(f, grad_f, B, p):
    return f + np.dot(grad_f, p) + 0.5 * np.dot(p, np.dot(B, p))


def dogleg_method(f, grad_f, x0, delta_hat, tol=1e-6, delta=0.5, eta=0.15):
    x = np.array(x0)
    k = 0
    f_values = []  # value of f at each iteration

    while np.linalg.norm(grad_f(x)) > tol:
        g = grad_f(x)
        B = hessian_approx(x, grad_f)

        p_B = -np.dot(np.linalg.inv(B), g)  # Newton step
        if np.linalg.norm(p_B) <= delta:
            p = p_B
        else:
            p_U = -np.dot(g, g) / np.dot(g, np.dot(B, g)) * g  # Steepest descent step
            if np.linalg.norm(p_U) >= delta:
                p = delta / np.linalg.norm(p_U) * p_U
            else:
                # Dogleg step
                a = np.dot(p_B - p_U, p_B - p_U)
                b = 2 * np.dot(p_B - p_U, p_U)
                c = np.dot(p_U, p_U) - delta**2
                # solving scalar quadratic on page 75 of Nocedal
                tau = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
                p = p_U + tau * (p_B - p_U)

        rho = (f(x) - f(x + p)) / (m_k(f(x), grad_f(x), B, [0, 0]) - m_k(f(x), grad_f(x), B, p))

        if rho < 0.25:
            delta = 0.25 * delta
        elif rho > 0.75 and np.linalg.norm(p) == delta:
            delta = min(2 * delta, delta_hat)

        if rho > eta:
            x = x + p

        k += 1
        f_values.append(f(x))

    return x, k, f_values


def rtrm(f, grad_f, x_0, delta_0=0.5, eta_1=0.15, eta_tilde_1=0.9,
         eta_tilde_2=0.9, gamma_1=0.9, gamma_2=0.9, tol=1e-6):
    x = x_0
    delta_k = delta_0
    k = 0
    f_values = []  # value of f at each iteration
    x_prev = None
    delta_prev = None

    while np.linalg.norm(grad_f(x)) > tol:
        # Step 1: Model definition
        fx = f(x)
        grad_fx_k = grad_f(x)
        B_k = hessian_approx(x, grad_f)

        # Step 2: Retrospective trust-region radius update
        if k > 0:
            if np.all(x == x_prev):
                # delta_k = np.random.uniform(gamma_1 * delta_prev, gamma_2 * delta_prev)
                delta_k = gamma_1 * delta_prev
            else:
                rho_tilde_k = (f(x_prev) - fx) / \
                              (m_k(f(x_prev), grad_f(x_prev), B_k, x_prev - x)
                               - m_k(fx, grad_fx_k, B_k, x_prev - x))
                if rho_tilde_k >= eta_tilde_2:
                    delta_k = delta_prev
                elif eta_tilde_1 <= rho_tilde_k < eta_tilde_2:
                    delta_k = gamma_2 * delta_prev
                else:
                    delta_k = gamma_1 * delta_prev

        # Step 3: Step calculation
        s_k = -delta_k * grad_fx_k / np.linalg.norm(grad_fx_k)

        # Step 4: Acceptance of the trial point
        rho_k = (fx - f(x + s_k)) / \
                (m_k(fx, grad_fx_k, B_k, np.zeros_like(x))
                 - m_k(fx, grad_fx_k, B_k, s_k))
        if rho_k > eta_1:
            x_next = x + s_k
        else:
            x_next = x

        x_prev = x
        x = x_next
        delta_prev = delta_k
        k += 1
        f_values.append(f(x))

    return x, k, f_values


x0 = np.array([3.0, -2.9])  # Initial point
delta_bound = 1
result_dog_b, num_dog_b, values_dog_b = dogleg_method(rosenbrock,
                                                      rosenbrock_gradient, x0, delta_bound)
result_dog_s, num_dog_s, values_dog_s = dogleg_method(sphere,
                                                      sphere_gradient, x0, delta_bound)
result_dog_m, num_dog_m, values_dog_m = dogleg_method(mccormick,
                                                      mccormick_gradient, x0, delta_bound)
print("Optimal point for Rosenbrock using Dogleg:", result_dog_b)
print("Number of iterations for Rosenbrock using Dogleg:", num_dog_b)
print("Optimal point for Sphere using Dogleg:", result_dog_s)
print("Number of iterations for Sphere using Dogleg:", num_dog_s)
print("Optimal point for McCormick using Dogleg:", result_dog_m)
print("Number of iterations for McCormick using Dogleg:", num_dog_m)

print("------------------------------------------")

result_rtrm_b, num_rtrm_b, values_rtrm_b = rtrm(rosenbrock, rosenbrock_gradient, x0)
result_rtrm_s, num_rtrm_s, values_rtrm_s = rtrm(sphere, sphere_gradient, x0)
result_rtrm_m, num_rtrm_m, values_rtrm_m = rtrm(mccormick, mccormick_gradient, x0)
print("Optimal point for Rosenbrock using RTRM:", result_rtrm_b)
print("Number of iterations for Rosenbrock using RTRM:", num_rtrm_b)
print("Optimal point for Sphere using RTRM:", result_rtrm_s)
print("Number of iterations for Sphere using RTRM:", num_rtrm_s)
print("Optimal point for McCormick using RTRM:", result_rtrm_m)
print("Number of iterations for McCormick using RTRM:", num_rtrm_m)

# comparing convergence rates for each function

# Rosenbrock
plt.figure(1)
plt.plot(np.arange(1, num_dog_b+1), values_dog_b, label='Dogleg', marker='.', color='red', alpha=0.5)
plt.plot(np.arange(1, num_rtrm_b+1), values_rtrm_b, label='RTRM', marker='.', color='blue', alpha=0.5)
plt.axhline(y=0, color='black', linestyle='--', label='Minimum')
plt.xlabel('Iterations')
plt.ylabel('Objective Function Value')
plt.xlim(0, 100)
plt.title('Convergence Rates of RTRM and Dogleg Method for Rosenbrock')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('rosenbrock comparison.png')
plt.show()


# Sphere
plt.figure(2)
plt.plot(np.arange(1, num_dog_s+1), values_dog_s, label='Dogleg', marker='.', color='red', alpha=0.5)
plt.plot(np.arange(1, num_rtrm_s+1), values_rtrm_s, label='RTRM', marker='.', color='blue', alpha=0.5)
plt.axhline(y=0, color='black', linestyle='--', label='Minimum')
plt.xlabel('Iterations')
plt.ylabel('Objective Function Value')
plt.xlim(0, 100)
plt.title('Convergence Rates of RTRM and Dogleg Method for Sphere')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('sphere comparison.png')
plt.show()


# McCormick
plt.figure(3)
plt.plot(np.arange(1, num_dog_m+1), values_dog_m, label='Dogleg', marker='.', color='red', alpha=0.5)
plt.plot(np.arange(1, num_rtrm_m+1), values_rtrm_m, label='RTRM', marker='.', color='blue', alpha=0.5)
plt.axhline(y=-1.9133, color='black', linestyle='--', label='Minimum')
plt.xlabel('Iterations')
plt.ylabel('Objective Function Value')
plt.xlim(0, 100)
plt.title('Convergence Rates of RTRM and Dogleg Method for McCormick')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('mccormick comparison.png')
plt.show()




