import matplotlib.pyplot as plt
import numpy as np
from scipy import sign
from cvxopt import solvers, matrix
from multiprocessing import Pool, cpu_count

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
    solvers.options['show_progress'] = False
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

    '''
    # Traditional QP: convert equation constrains Ax=b into "Ax>=b and -Ax>=-b"
    G_original = np.identity(N) * -1
    G_plus = np.zeros((N+2, N))
    G_plus[:N, :] = G_original
    G_plus[N, :] = y
    G_plus[N+1, :] = -1 * y
    G = matrix(G_plus)
    h = matrix(0.0, (N+2, 1))
    '''

    solvers.options['show_progress'] = False
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


def get_error(target_w, target_b, model_w, model_b):
    def get_x1(x0, w, b):
        return -1 * (w[0] * x0 + b) / w[1]
    target_x1_pos = get_x1(1, target_w, target_b)
    target_x1_minus = get_x1(-1, target_w, target_b)
    model_x1_pos = get_x1(1, model_w, model_b)
    model_x1_minus = get_x1(-1, model_w, model_b)
    error = abs(target_x1_pos - model_x1_pos) + abs(target_x1_minus - model_x1_minus)
    return error


def get_error_monte_carlo(target_w, target_b, pla_w, pla_b, svm_w, svm_b, N=10000):
    x = np.random.uniform(-1, 1, (N, 2))
    s = np.dot(x, target_w) + target_b
    y = sign(s)
    pla_s = np.dot(x, pla_w) + pla_b
    pla_y = sign(pla_s)
    svm_s = np.dot(x, svm_w) + svm_b
    svm_y = sign(svm_s)
    return len(pla_y[pla_y != y]), len(svm_y[svm_y != y])


def compare_pla_svm(N=10):
    target_w, target_b = generate_target_weight_bias()
    x, y = generate_samples(N, target_w, target_b)
    pla_w, pla_b = get_pla_weight_bias(x, y)
    svm_w, svm_b, sv = svm_hard_margin_dual(x, y)
    #pla_error = get_error(target_w, target_b, pla_w, pla_b)
    #svm_error = get_error(target_w, target_b, svm_w, svm_b)
    pla_error, svm_error = get_error_monte_carlo(target_w, target_b, pla_w, pla_b, svm_w, svm_b)
    svm_win = 0
    if pla_error > svm_error:
        svm_win = 1
    return [svm_win, len(sv)]


def get_svm_wins(N=10, M=1000):
    svm_wins = 0.0
    num_sv = 0.0
    for i in range(M):
        svm_win, sv = compare_pla_svm(N)
        svm_wins += svm_win / M
        num_sv += sv / M
    return [svm_wins, num_sv]


def get_svm_wins_multiprocessing(N=10, M=1000):
    # python quit unexpectedly on macos for N=100, M=1000
    args = [N for i in range(M)]
    pool = Pool(cpu_count())
    results = pool.map(compare_pla_svm, args)
    #a = sum(row[0] for row in results)
    #b = sum(row[1] for row in results)
    #return [a/M, b/M]
    results = np.array(pool.map(compare_pla_svm, args)) # python quit not due to numpy
    return np.sum(results, axis=0) / M


def unit_test(N=10):
    target_w, target_b = generate_target_weight_bias()
    x, y = generate_samples(N, target_w, target_b)
    print_sample(x, y)
    wbline(target_w, target_b)

    pla_w, pla_b = get_pla_weight_bias(x, y)
    wbline(pla_w, pla_b, fmt=":")

    # svm_w, svm_b = svm_hard_margin_primal(x, y)
    svm_w, svm_b, sv = svm_hard_margin_dual(x, y)
    wbline(svm_w, svm_b, fmt="--")
    print_support_vector(x, y, sv)
    print("support vectors:", sv)

    # pla_error = get_error(target_w, target_b, pla_w, pla_b)
    # svm_error = get_error(target_w, target_b, svm_w, svm_b)
    pla_error, svm_error = get_error_monte_carlo(target_w, target_b, pla_w, pla_b, svm_w, svm_b)

    print("pla_error = %.3f" % pla_error)
    print("svm_error = %.3f" % svm_error)

    print("cpu_count=", cpu_count())

    plt.show()


if __name__ == '__main__':
    #unit_test()

    svm_wins_10, num_sv_10 = get_svm_wins(N=10)
    svm_wins_100, num_sv_100 = get_svm_wins(N=100)
    print("N=10, svm_wins=%.3f, num_sv=%.2f" % (svm_wins_10, num_sv_10))
    print("N=100, svm_wins=%.3f, num_sv=%.2f" % (svm_wins_100, num_sv_100))
