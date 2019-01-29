import numpy as np
from scipy import exp


class LogisticRegression:
    def __init__(self, train_file, test_file):
        self.train_data = np.loadtxt(train_file)
        self.train_x = np.ones(self.train_data.shape)
        self.train_x[:, :-1] = self.train_data[:, :-1]
        self.train_y = self.train_data[:, -1]
        self.test_data = np.loadtxt(test_file)
        self.test_x = np.ones(self.test_data.shape)
        self.test_x[:, :-1] = self.test_data[:, :-1]
        self.test_y = self.test_data[:, -1]
        self.weight = np.ones(self.train_data.shape[1])

    def sigmoid(self, s):
        return 1 / (1 + exp(-1*s))

    def gradient(self):
        a = self.sigmoid(-1 * self.train_x.dot(self.weight) * self.train_y)
        b = -1 * self.train_x * self.train_y[:, np.newaxis]
        return np.mean(b * a[:, np.newaxis], axis=0)

    def gradient_descent(self, eta=0.001, T=2000):
        counter = 0
        while counter < T:
            self.weight -= eta * self.gradient()
            counter += 1

    def single_gradient(self, i):
        a = self.sigmoid(-1 * self.train_x[i].dot(self.weight) * self.train_y[i])
        b = -1 * self.train_x[i] * self.train_y[i]
        return b * a

    def sgd(self, eta=0.001, T=2000):
        counter = 0
        N = len(self.train_y)
        while counter < T:
            self.weight -= eta * self.single_gradient(counter % N)
            counter += 1

    def get_error_rate(self):
        prediction = self.sigmoid(self.test_x.dot(self.weight))
        prediction[ prediction > 0.5 ] = 1
        prediction[ prediction != 1 ] = -1
        error = np.zeros(self.test_y.shape)
        error[ self.test_y != prediction ] = 1
        error_rate = error.sum() / len(error)
        return error_rate

    def reset_weight(self):
        self.weight = np.zeros(self.weight.shape)


if __name__ == '__main__':
    TRAIN_FILE = "./hw3_train.dat"
    TEST_FILE = "./hw3_test.dat"

    model = LogisticRegression(TRAIN_FILE, TEST_FILE)
    model.gradient_descent(eta=0.001)
    gd_e_out = model.get_error_rate()
    print("eta = 0.001, gd_e_out = %.3f" % gd_e_out)

    model.reset_weight()
    model.gradient_descent(eta=0.01)
    gd_e_out = model.get_error_rate()
    print("eta = 0.01, gd_e_out = %.3f" % gd_e_out)

    model.reset_weight()
    model.sgd(eta=0.001)
    sgd_e_out = model.get_error_rate()
    print("eta = 0.001, sgd_e_out = %.3f" % sgd_e_out)

    model.reset_weight()
    model.sgd(eta=0.01)
    sgd_e_out = model.get_error_rate()
    print("eta = 0.01, sgd_e_out = %.3f" % sgd_e_out)
