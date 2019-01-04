import random
from scipy import stats, sign


def sample_generator(N=20, noise=0.2):
    while True:
        random.seed()
        x = []
        y = []
        for i in range(N):
            x.append(random.uniform(-1, 1))
            if stats.bernoulli.rvs(noise, size=1) == 1:
                y.append(-1 * sign(x[i]))
            else:
                y.append(sign(x[i]))
        yield x, y


def get_e_in(theta, x, y):
    e_in = 0
    for i in range(len(x)):
        if (x[i] <= theta and y[i] == 1) or (x[i] > theta and y[i] == -1):
            e_in += 1 / len(x)
    return e_in


def get_e_out(theta):
    e_out = 0.5 + 0.3 * (abs(theta) - 1)
    return e_out


def decision_stump(x, y):
    optimal_e_in = 1.0
    optimal_theta = 0.0
    for i in range(len(x)+1):
        if i == len(x):
            theta = -1
        else:
            theta = x[i] + 1e-8
        e_in = get_e_in(theta, x, y)
        if e_in < optimal_e_in:
            optimal_e_in = e_in
            optimal_theta = theta
    e_out = get_e_out(optimal_theta)
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





