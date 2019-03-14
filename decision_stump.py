import numpy as np
from scipy import sign
import sys


def sample_generator(N=20, noise=0.2):
    while True:
        x = np.random.uniform(-1, 1, N)
        noise_flag = np.random.binomial(1, noise, N)
        y = sign(x)
        y[noise_flag == 1] *= -1
        p = x.argsort()
        yield x[p], y[p]


def get_e_in(theta, s, x, y, u=None):
    e_in = 0
    correct = []
    incorrect = []
    for i in range(len(x)):
        if s * sign(x[i] - theta) != y[i]:
            incorrect.append(i)
            if u is None:
                e_in += 1 / len(x)
            else:
                e_in += u[i] / len(x)
        else:
            correct.append(i)
    return e_in, correct, incorrect


def get_e_out(theta, s, noise=0.2):
    e_out = 0.5 + (0.5-noise) * s * (abs(theta) - 1)
    return e_out


def decision_stump(x, y, u=None):
    sorted_x = np.sort(x)
    optimal_e_in = 1.0
    optimal_theta = 0.0
    optimal_s = 1
    optimal_correct = []
    optimal_incorrect = []
    for i in range(len(sorted_x)):
        if i == 0:
            theta = float("-inf")
        else:
            theta = (x[i-1] + x[i]) / 2
        e_in_positive, correct_positive, incorrect_positive = get_e_in(theta, 1, x, y, u)
        e_in_negative, correct_negative, incorrect_negative = get_e_in(theta, -1, x, y, u)
        if e_in_positive <= e_in_negative:
            e_in = e_in_positive
            s = 1
            correct = correct_positive
            incorrect = incorrect_positive
        else:
            e_in = e_in_negative
            s = -1
            correct = correct_negative
            incorrect = incorrect_negative
        if e_in < optimal_e_in:
            optimal_e_in = e_in
            optimal_theta = theta
            optimal_s = s
            optimal_correct = correct
            optimal_incorrect = incorrect
    e_out = get_e_out(optimal_theta, optimal_s)
    return optimal_e_in, e_out, optimal_theta, optimal_s, optimal_correct, optimal_incorrect


def get_average_errors(num_experiments=5000):
    e_in_average = 0.0
    e_out_average = 0.0
    gen = sample_generator()
    for i in range(num_experiments):
        x, y = gen.__next__()
        e_in, e_out, _, _, _, _ = decision_stump(x, y)
        e_in_average += e_in / num_experiments
        e_out_average += e_out / num_experiments
    return e_in_average, e_out_average


def multi_dim_decision_stump(x, y, u):
    optimal_theta = 0.0
    optimal_s = 1
    optimal_i = 0
    optimal_e_in = 1.0
    optimal_correct = []
    optimal_incorrect = []

    for i in range(x.shape[1]):
        e_in, _, theta, s, correct, incorrect = decision_stump(x[:, i], y, u)
        if e_in <= optimal_e_in:
            optimal_e_in = e_in
            optimal_theta = theta
            optimal_s = s
            optimal_i = i
            optimal_correct = correct
            optimal_incorrect = incorrect

    return optimal_e_in, optimal_theta, optimal_s, optimal_i, optimal_correct, optimal_incorrect


def get_data(file):
    data = np.loadtxt(file)
    x = data[:, :-1]
    y = data[:, -1]
    return x, y


def get_e_out_multi_dim_ds(theta, s, i, x, y):
    return get_e_in(theta, s, x[:, i], y)


def get_multi_dim_ds_errors(train_file, test_file):
    train_x, train_y = get_data(train_file)
    test_x, test_y = get_data(test_file)
    e_in, theta, s, i, _, _ = multi_dim_decision_stump(train_x, train_y)
    e_out = get_e_out_multi_dim_ds(theta, s, i, test_x, test_y)
    return e_in, e_out


if __name__ == '__main__':
    if sys.argv[1] == "single_dim_ds":
        e_in_average, e_out_average = get_average_errors()
        print("average Ein = %.3f" % e_in_average)
        print("average Eout = %.3f" % e_out_average)

    if sys.argv[1] == "multi_dim_ds":
        TRAIN_FILE = "./hw2_train.dat"
        TEST_FILE = "./hw2_test.dat"
        e_in, e_out = get_multi_dim_ds_errors(TRAIN_FILE, TEST_FILE)
        print("multi dim ds Ein = %.3f" % e_in)
        print("multi dim ds Eout = %.3f" % e_out)
