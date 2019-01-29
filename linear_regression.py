import matplotlib.pyplot as plt
import numpy as np
from scipy import sign


def generate_sample(N=1000, noise=0.1):
    x = np.random.uniform(-1, 1, (N, 2))
    y = sign(pow(x[:,0], 2) + pow(x[:,1], 2) - 0.6)
    noise_flag = np.random.binomial(1, noise, N)
    y[noise_flag == 1] *= -1
    return x, y


def add_x0(x):
    x_plus = np.ones((x.shape[0], x.shape[1] + 1))
    x_plus[:, 1:] = x
    return x_plus


def linear_regression(x, y, pseudo_inverse=True):
    if pseudo_inverse is True:
        weight = np.linalg.pinv(x).dot(y)
    else:
        weight = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return weight


def transform_to_z(x):
    z = np.ones((x.shape[0], 6))
    z[:, :3] = x
    z[:, 3] = x[:, 1] * x[:, 2]
    z[:, 4] = x[:, 1] ** 2
    z[:, 5] = x[:, 2] ** 2
    return z


def get_error_rate(w_lin, x, y):
    y_hat = sign(np.dot(x, w_lin))
    error = np.zeros(y.shape)
    error[y != y_hat] = 1
    error_rate = error.sum() / len(error)
    return error_rate


def get_avg_error_rate(num_experiments=1000, transform=False):
    counter = 0
    avg_e_in = 0.0
    avg_e_out = 0.0
    while counter < num_experiments:
        train_x, train_y = generate_sample()
        test_x, test_y = generate_sample()
        train_x = add_x0(train_x)
        test_x = add_x0(test_x)
        if transform is True:
            train_x = transform_to_z(train_x)
            test_x = transform_to_z(test_x)

        w_lin = linear_regression(train_x, train_y, pseudo_inverse=True)
        e_in = get_error_rate(w_lin, train_x, train_y)
        e_out = get_error_rate(w_lin, test_x, test_y)
        avg_e_in += e_in / num_experiments
        avg_e_out += e_out / num_experiments

        counter += 1
    return avg_e_in, avg_e_out


def print_sample(x, y):
    plt.figure(figsize=(8, 8))
    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(x[i,0], x[i,1], c='r', marker='x')
        else:
            plt.scatter(x[i,0], x[i,1], c='b', marker='o')


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', c='k')


if __name__ == '__main__':
    x_avg_e_in, x_avg_e_out = get_avg_error_rate()
    print("x_avg_e_in = %.3f, x_avg_e_out = %.3f" % (x_avg_e_in, x_avg_e_out))
    z_avg_e_in, z_avg_e_out = get_avg_error_rate(transform=True)
    print("z_avg_e_in = %.3f, z_avg_e_out = %.3f" % (z_avg_e_in, z_avg_e_out))