import numpy as np
import matplotlib.pyplot as plt


def load_samples(sample_file):
    samples = np.loadtxt(sample_file)
    x = np.ones(samples.shape)
    x[:, 1:] = samples[:, 1:]  # x = [1, x0, x1]
    y = samples[:, 0]
    return x, y


def plot_samples(x, y):
    plt.figure(figsize=(8, 8))
    plt.scatter(x[y>3][:,1], x[y>3][:,2], c=(1/(y[y>3]+10)), marker='x')


def unit_test():
    train_file = "./features.train"
    test_file = "./features.test"
    train_x, train_y = load_samples(train_file)
    test_x, test_y = load_samples(test_file)
    #plot_samples(train_x, train_y)
    plot_samples(test_x, test_y)
    plt.show()


if __name__ == '__main__':
    unit_test()
