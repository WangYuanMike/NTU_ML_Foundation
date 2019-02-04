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


def read_sample(sample_file):
    sample = np.loadtxt(sample_file)
    x = np.ones(sample.shape)
    x[:, :-1] = sample[:, :-1]
    y = sample[:, -1]
    return x, y


def ridge_regression(x, y, LAMBDA=10):
    """linear regression with 2 norm regularization"""
    weight = np.linalg.inv(LAMBDA * np.identity(x.shape[1]) + x.T.dot(x)).dot(x.T).dot(y)
    return weight


def transform_to_z(x):
    z = np.ones((x.shape[0], 6))
    z[:, :3] = x
    z[:, 3] = x[:, 1] * x[:, 2]
    z[:, 4] = x[:, 1] ** 2
    z[:, 5] = x[:, 2] ** 2
    return z


def get_error_rate(weight, x, y):
    y_hat = sign(np.dot(x, weight))
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
    # Linear regression
    x_avg_e_in, x_avg_e_out = get_avg_error_rate()
    print("x_avg_e_in = %.3f, x_avg_e_out = %.3f" % (x_avg_e_in, x_avg_e_out))
    print()

    # Linear regression with non-linear transformation
    z_avg_e_in, z_avg_e_out = get_avg_error_rate(transform=True)
    print("z_avg_e_in = %.3f, z_avg_e_out = %.3f" % (z_avg_e_in, z_avg_e_out))
    print()

    # Ridge regression
    train_x, train_y = read_sample("./hw4_train.dat")
    w_reg = ridge_regression(train_x, train_y)
    e_in_reg = get_error_rate(w_reg, train_x, train_y)
    test_x, test_y = read_sample("./hw4_test.dat")
    e_out_reg = get_error_rate(w_reg, test_x, test_y)
    print("e_in_reg = %.3f" % e_in_reg)
    print("e_out_reg = %.3f" % e_out_reg)
    print()

    # Ridge regression without validation
    lamb = 1e2
    while lamb >= 1e-10:
        w_reg = ridge_regression(train_x, train_y, LAMBDA=lamb)
        e_in_reg = get_error_rate(w_reg, train_x, train_y)
        e_out_reg = get_error_rate(w_reg, test_x, test_y)
        print("lambda = %.2e, e_in = %.3f, e_out = %.3f" % (lamb, e_in_reg, e_out_reg))
        lamb /= 10
    print()

    # Ridge regression with validation
    train_x_minus = train_x[:120]
    train_y_minus = train_y[:120]
    val_x = train_x[120:]
    val_y = train_y[120:]
    lamb = 1e2
    while lamb >= 1e-10:
        w_reg = ridge_regression(train_x_minus, train_y_minus, LAMBDA=lamb)
        e_train = get_error_rate(w_reg, train_x_minus, train_y_minus)
        e_val = get_error_rate(w_reg, val_x, val_y)
        e_test = get_error_rate(w_reg, test_x, test_y)
        print("lambda = %.2e, e_train = %.3f, e_val = %.3f, e_test = %.3f" %
              (lamb, e_train, e_val, e_test))
        lamb /= 10
    print()

    # Ridge regression with optimal lambda found by validation
    w_reg = ridge_regression(train_x, train_y, LAMBDA=1e0)
    e_in = get_error_rate(w_reg, train_x, train_y)
    e_out = get_error_rate(w_reg, test_x, test_y)
    print("lambda = %.2e, e_in = %.3f, e_out = %.3f" % (lamb, e_in, e_out))
    print()

    # Ridge regression with 5-fold cross validation
    e_cv = np.zeros(13)
    for i in range(5):
        train_x_minus = np.delete(train_x, np.s_[40*i : 40*(i+1)], axis=0)
        train_y_minus = np.delete(train_y, np.s_[40*i : 40*(i+1)], axis=0)
        print(train_x_minus.shape)
        val_x = train_x[40*i : 40*(i+1)]
        val_y = train_y[40*i : 40*(i+1)]

        lamb = 1e2
        index = 0
        while lamb >= 1e-10:
            w_reg = ridge_regression(train_x_minus, train_y_minus, LAMBDA=lamb)
            e_cv[index] += get_error_rate(w_reg, val_x, val_y) / 5
            index += 1
            lamb /= 10

        print("e_cv = ", e_cv)

    lamb = 1e2
    index = 0
    while lamb >= 1e-10:
        print("lambda = %.2e, e_cv = %.3f" % (lamb, e_cv[index]))
        index += 1
        lamb /= 10
    print()

    # Ridge regression with optimal lambda found by 5-fold cross validation
    w_reg = ridge_regression(train_x, train_y, LAMBDA=1e-8)
    e_in = get_error_rate(w_reg, train_x, train_y)
    e_out = get_error_rate(w_reg, test_x, test_y)
    print("lambda = %.2e, e_in = %.3f, e_out = %.3f" % (lamb, e_in, e_out))
    print()