import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cvxopt import solvers, matrix

import PLA_SVM as ps


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
    sv = np.squeeze(np.where(alpha > 1e-3))
    free_sv = np.squeeze(np.where((alpha > 1e-3) & (alpha / c < 0.99)))
    weight = np.squeeze((alpha[sv][:, None] * y[sv][:, None]).T.dot(x[sv]))
    bias = y[free_sv[0]] - np.dot(weight, x[free_sv[0]])
    print(alpha)
    print(sv)
    print(free_sv)
    print(weight)
    print(bias)
    return alpha, sv, free_sv, weight, bias


def unit_test():
    N = 10
    target_w, target_b = ps.generate_target_weight_bias()
    x, y = ps.generate_samples(N, target_w, target_b)
    ps.print_sample(x, y)
    ps.wbline(target_w, target_b)

    pla_w, pla_b = ps.get_pla_weight_bias(x, y)
    ps.wbline(pla_w, pla_b, fmt=":")

    _, sv, free_sv, svm_w, svm_b = svm_soft_margin_dual(x, y, c=10)
    ps.wbline(svm_w, svm_b, fmt="--")
    ps.print_support_vector(x, sv, free_sv)

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
