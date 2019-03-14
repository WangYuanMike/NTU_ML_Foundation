import kernel_ridge_regression as krr
import numpy as np


def bootstrap(x, y):
    index = list(range(len(x)))
    a = np.random.choice(index, len(x), replace=True)
    return x[a], y[a]


def bagging(T, lamb, train_x, train_y, test_x, test_y):
    total_y_in = np.zeros(len(train_y))
    total_y_out = np.zeros(len(test_y))
    for t in range(T):
        boot_train_x, boot_train_y = bootstrap(train_x, train_y)
        model = krr.KernelRidgeRegression(kernel='l', lamb=lamb)
        model.fit(boot_train_x, boot_train_y)
        predict_y_in, _ = model.predict(train_x, train_y)
        predict_y_out, _ = model.predict(test_x, test_y)
        total_y_in += predict_y_in
        total_y_out += predict_y_out
    bagging_y_in = np.sign(total_y_in)
    bagging_y_out = np.sign(total_y_out)
    return bagging_y_in, bagging_y_out, krr.get_error(train_y, bagging_y_in), krr.get_error(test_y, bagging_y_out)


def bagging_krr_experiment(T):
    x, y = krr.load_samples("./hw2_lssvm_all.dat")
    train_x = x[:400, :]
    train_y = y[:400]
    test_x = x[400:, :]
    test_y = y[400:]

    lamb_list = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
    e_in_list = np.zeros(len(lamb_list))
    e_out_list = np.zeros(e_in_list.shape)
    for i, lamb in enumerate(lamb_list):
        _, _, e_in, e_out = bagging(T, lamb, train_x, train_y, test_x, test_y)
        e_in_list[i] = e_in
        e_out_list[i] = e_out

    print("e_in_list:")
    print(e_in_list)
    print("e_out_list")
    print(e_out_list)


if __name__ == '__main__':
    krr.linear_krr_experiment()
    bagging_krr_experiment(250)
