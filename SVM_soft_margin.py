import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cvxopt import solvers, matrix
from scipy import exp

import PLA_SVM as ps

EPSILON = 1e-4

def load_samples(sample_file):
    samples = np.loadtxt(sample_file)
    x = np.ones(samples.shape)
    x[:, 1:] = samples[:, 1:]  # x = [1, x0, x1]
    y = samples[:, 0]
    return x, y


def plot_samples(x, y):
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(x[:,1], x[:,2], s=32, c=y, cmap=cm.get_cmap('Set1'), marker='*')
    plt.colorbar(sc)


def svm_soft_margin_dual(x, y, c, kernel=None, q=1, zeta=0, gamma=1):
    def polynomial_kernel(x, q, zeta, gamma):
        return np.power(zeta + gamma * np.dot(x, x.T), q)

    def gaussian_kernel(x, gamma):
        x_norm = np.sum(x ** 2, axis=-1)
        return np.exp(-gamma * (x_norm[:, None] + x_norm[None, :] - 2 * np.dot(x, x.T)))

    N = len(y)

    if kernel == 'p':
        k = polynomial_kernel(x, q, zeta, gamma)
    elif kernel == 'g':
        k = gaussian_kernel(x, gamma)
    else:
        k = polynomial_kernel(x, 1, 0, 1)   # linear kernel

    Q = matrix(np.outer(y, y) * k)
    p = matrix(-1.0, (N, 1))
    G_0 = np.identity(N) * -1
    G_c = np.identity(N)
    G_np = np.zeros((2 * N, N))
    G_np[:N, :] = G_0
    G_np[N:, :] = G_c
    G = matrix(G_np)
    h_np = np.zeros((2 * N, 1))
    h_np[N:] += c
    h = matrix(h_np)
    A = matrix(y[None, :])
    b = matrix(0.0)
    solvers.options['show_progress'] = False
    sol = solvers.qp(Q, p, G, h, A, b)

    alpha = np.squeeze(sol['x'])
    sv = np.squeeze(np.where(alpha > EPSILON))
    free_sv = np.squeeze(np.where((alpha > EPSILON) & (alpha / c < 0.99)))
    return alpha, sv, free_sv


def kernel_score(x, y, sv, free_sv, alpha, x_prime, kernel=None, q=1, zeta=0, gamma=1):
    def polynomial_kernel(x, x_prime, q=1, zeta=0, gamma=1):
        return pow(zeta + gamma * np.dot(x, x_prime), q)

    def gaussian_kernel(x, x_prime, gamma=1):
        return exp(-1 * np.linalg.norm(x - x_prime, 2) ** 2)

    def kernel_sum(x, y, sv, alpha, x_prime, kernel):
        if kernel is None or kernel == 'p':
            sum = np.dot(np.squeeze(alpha[sv][:, None] * y[sv][:, None]),
                         polynomial_kernel(x[sv], x_prime, q, zeta, gamma))
        elif kernel == 'g':
            sum = np.dot(np.squeeze(alpha[sv][:, None] * y[sv][:, None]),
                         gaussian_kernel(x[sv], x_prime, gamma))
        return sum

    b = y[free_sv[0]] - kernel_sum(x, y, sv, alpha, x[free_sv[0]], kernel)
    return kernel_sum(x, y, sv, alpha, x_prime, kernel) + b


def draw_decision_boundary(x, y, sv, free_sv, alpha):
    offset = 1e-2
    x1 = np.arange(-1.0, 1.0, offset)
    x2 = x1
    x_print = np.zeros((x1.shape[0], 2))

    for index, i in enumerate(x1):
        for j in x2:
            x_prime = np.array([i, j])
            a = abs(kernel_score(x, y, sv, free_sv, alpha, x_prime))
            print(a)
            if a < 1e-2:
                x_print[index] = x_prime
                print(x_prime)
                break

    print(x_print)
    plt.scatter(x_print[:, 0], x_print[:, 1], marker='.')


def unit_test():
    N = 10
    target_w, target_b = ps.generate_target_weight_bias()
    x, y = ps.generate_samples(N, target_w, target_b)
    ps.print_sample(x, y)
    ps.wbline(target_w, target_b)

    pla_w, pla_b = ps.get_pla_weight_bias(x, y)
    ps.wbline(pla_w, pla_b, fmt=":")

    alpha, sv, free_sv = svm_soft_margin_dual(x, y, c=10)
    svm_w = np.squeeze((alpha[sv][:, None] * y[sv][:, None]).T.dot(x[sv]))
    svm_b = y[free_sv[0]] - np.dot(svm_w, x[free_sv[0]])
    ps.wbline(svm_w, svm_b, fmt="--")
    ps.print_support_vector(x, sv, free_sv)

    draw_decision_boundary(x, y, sv, free_sv, alpha)

    plt.show()

    '''
    train_file = "./features.train"
    test_file = "./features.test"
    train_x, train_y = load_samples(train_file)
    test_x, test_y = load_samples(test_file)
    plot_samples(train_x, train_y)
    #plot_samples(test_x, test_y)
    plt.show()
    '''


if __name__ == '__main__':
    unit_test()
