import numpy as np
from sklearn_SVM import get_error


class KernelRidgeRegression():
    def __init__(self, kernel='g', gamma=1.0, lamb=1.0):
        self.kernel = kernel
        self.gamma = gamma
        self.lamb = lamb
        self.beta = None
        self.train_x = None

    def fit(self, x, y):
        def linear_kernel():
            return np.dot(x, x.T)

        def rbf_kernel():
            ns = np.linalg.norm(x, axis=1) ** 2
            return np.exp(-self.gamma * (ns[:, None] + ns[None, :] - 2 * np.dot(x, x.T)))

        if self.kernel == 'g':
            k = rbf_kernel()
        else:
            k = linear_kernel()

        self.beta = np.dot(np.linalg.inv(self.lamb * np.identity(len(y)) + k), y)
        self.train_x = x

    def predict(self, x, y):
        def linear_kernel():
            return np.dot(x, self.train_x.T)

        def rbf_kernel():
            return np.exp(-self.gamma * np.linalg.norm(self.train_x[None, :, :] - x[:, None, :], axis=2) ** 2)

        if self.kernel == 'g':
            k = rbf_kernel()
        else:
            k = linear_kernel()

        predict_y = np.sign(np.dot(k, self.beta))
        #print(predict_y)
        #print(y)
        error = get_error(y, predict_y)
        return predict_y, error


def load_samples(sample_file):
    samples = np.loadtxt(sample_file)
    x = samples[:, :-1]
    y = samples[:, -1]
    return x, y


def krr_experiment():
    x, y = load_samples("./hw2_lssvm_all.dat")
    train_x = x[:400, :]
    train_y = y[:400]
    test_x = x[400:, :]
    test_y = y[400:]

    gamma_list = [32, 2, 0.125]
    lamb_list = [1e-3, 1e0, 1e3]
    e_in_list = np.zeros((len(gamma_list), len(lamb_list)))
    e_out_list = np.zeros(e_in_list.shape)
    for i, gamma in enumerate(gamma_list):
        for j, lamb in enumerate(lamb_list):
            model = KernelRidgeRegression(gamma=gamma, lamb=lamb)
            model.fit(train_x, train_y)
            _, e_in = model.predict(train_x, train_y)
            _, e_out = model.predict(test_x, test_y)
            e_in_list[i, j] = e_in
            e_out_list[i, j] = e_out

    print("e_in_list:")
    print(e_in_list)
    print("e_out_list")
    print(e_out_list)


def linear_krr_experiment():
    x, y = load_samples("./hw2_lssvm_all.dat")
    train_x = x[:400, :]
    train_y = y[:400]
    test_x = x[400:, :]
    test_y = y[400:]

    lamb_list = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
    e_in_list = np.zeros(len(lamb_list))
    e_out_list = np.zeros(e_in_list.shape)
    for i, lamb in enumerate(lamb_list):
        model = KernelRidgeRegression(kernel='l', lamb=lamb)
        model.fit(train_x, train_y)
        _, e_in = model.predict(train_x, train_y)
        _, e_out = model.predict(test_x, test_y)
        e_in_list[i] = e_in
        e_out_list[i] = e_out

    print("e_in_list:")
    print(e_in_list)
    print("e_out_list")
    print(e_out_list)


if __name__ == '__main__':
    #krr_experiment()
    linear_krr_experiment()
