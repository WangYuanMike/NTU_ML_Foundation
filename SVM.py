import matplotlib.pyplot as plt
import numpy as np
from scipy import sign
from cvxopt import solvers, matrix

from linear_regression import print_sample, wbline
from PLA import Perceptron


def generate_samples(N, w, b):
    while True:
        x = np.random.uniform(-1, 1, (N, 2))
        s = np.dot(x, w) + b
        y = sign(s)
        if N > len(y[y == 1]) > 0:
            return x, y


def generate_target_weight_bias():
    p = np.random.uniform(-1, 1, (2, 2))
    a = np.array([[p[0, 1], 1], [p[1, 1], 1]])
    b = np.array([p[0, 0], p[1, 0]])
    x = np.linalg.solve(a, b)
    weight = [1, x[0]]
    bias = x[1]
    return weight, bias


def get_pla_weight_bias(x, y):
    p = Perceptron()
    p.init_from_data(x, y)
    p.random_pla()
    weight = p.weight[:-1]
    bias = p.weight[-1]
    return weight, bias


def svm_hard_margin_primal(x, y):
    N = len(y)
    x_plus = np.ones((x.shape[0], x.shape[1]+1))
    x_plus[:, 1:] = x

    Q = matrix([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    p = matrix(0.0, (3, 1))
    G = matrix(-1 * x_plus * y[:, np.newaxis])
    h = matrix(-1.0, (N, 1))
    sol = solvers.qp(Q, p, G, h)

    weight = sol['x'][1:]
    bias = sol['x'][0]
    return weight, bias


def svm_hard_margin_dual(x, y):
    N = len(y)
    s = x * y[:, np.newaxis]

    Q = matrix(np.dot(s, s.T))
    p = matrix(-1.0, (N, 1))
    G = matrix(0.0, (N, N))
    G[::N+1] = -1.0  # set every N+1 element to -1.0 (matrix diagonal)
    h = matrix(0.0, (N, 1))
    A = matrix(y[np.newaxis, :])
    b = matrix(0.0)
    sol = solvers.qp(Q, p, G, h, A, b)

    alpha = np.squeeze(sol['x'])
    sv = np.squeeze(np.where(alpha > 1e-3))
    weight = np.squeeze((alpha[sv][:, np.newaxis] * y[sv][:, np.newaxis]).T.dot(x[sv]))
    bias = y[sv[0]] - np.dot(weight, x[sv[0]])
    return weight, bias, sv


def print_support_vector(x, y, sv):
    for i in range(len(y)):
        if i in sv:
            plt.scatter(x[i,0], x[i,1], c='w', marker='.')


if __name__ == '__main__':
    N = 10
    target_w, target_b = generate_target_weight_bias()
    x, y = generate_samples(N, target_w, target_b)
    print_sample(x, y)
    wbline(target_w, target_b)

    pla_w, pla_b = get_pla_weight_bias(x, y)
    wbline(pla_w, pla_b, fmt=":")

    #svm_w, svm_b = svm_hard_margin_primal(x, y)
    svm_w, svm_b, sv = svm_hard_margin_dual(x, y)
    wbline(svm_w, svm_b, fmt="--")
    print_support_vector(x, y, sv)
    print(sv)

    plt.show()
