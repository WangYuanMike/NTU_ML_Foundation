import numpy as np
from scipy import sign


def sample_generator(N=20, noise=0.2):
    while True:
        x = np.random.uniform(-1, 1, N)
        noise_flag = np.random.binomial(1, noise, N)
        y = sign(x)
        y[noise_flag == 1] *= -1
        p = x.argsort()
        yield x[p], y[p]


def get_e_in(theta, s, x, y):
    e_in = 0
    for i in range(len(x)):
        if s * sign(x[i] - theta) != y[i]:
            e_in += 1 / len(x)
    return e_in


def get_e_out(theta, s, noise=0.2):
    e_out = 0.5 + (0.5-noise) * s * (abs(theta) - 1)
    return e_out


def decision_stump(x, y):
    optimal_e_in = 1.0
    optimal_theta = 0.0
    for i in range(len(x)):
        if i == 0:
            theta = (-1 + x[i]) / 2
        else:
            theta = (x[i-1] + x[i]) / 2
        e_in_positive = get_e_in(theta, 1, x, y)
        e_in_negative = get_e_in(theta, -1, x, y)
        if e_in_positive <= e_in_negative:
            e_in = e_in_positive
            s = 1
        else:
            e_in = e_in_negative
            s = -1
        if e_in < optimal_e_in:
            optimal_e_in = e_in
            optimal_theta = theta
            optimal_s = s
    e_out = get_e_out(optimal_theta, optimal_s)
    return optimal_e_in, e_out


def get_average_errors(num_experiments=5000):
    e_in_average = 0.0
    e_out_average = 0.0
    gen = sample_generator()
    for i in range(num_experiments):
        x, y = gen.__next__()
        e_in, e_out = decision_stump(x, y)
        e_in_average += e_in / num_experiments
        e_out_average += e_out / num_experiments
    return e_in_average, e_out_average


if __name__ == '__main__':
    e_in_average, e_out_average = get_average_errors()
    print("average Ein = %.3f" % e_in_average)
    print("average Eout = %.3f" % e_out_average)
