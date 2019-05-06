import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sqrt

from libsvm.svmutil import *
from sklearn_SVM import load_samples, ova


def weight_versus_c():
    x, y = load_samples("./features.train")
    y_ova_0 = ova(y, 0)

    c_list = [1e-5, 1e-3, 1e-1, 1e1, 1e3]
    w_norm_list = []
    for c in c_list:
        arg_option = "-t 0 -c " + str(c)
        model = svm_train(y_ova_0, x, arg_option)
        sv_coef = np.squeeze(np.array(model.get_sv_coef()))
        sv = pd.DataFrame(model.get_SV()).values[:, 1:]
        w = np.dot(sv_coef, sv)
        w_norm_list.append(np.linalg.norm(w))

    print(w_norm_list)
    plt.scatter(np.log10(c_list), w_norm_list)
    plt.show()


def e_in_versus_c():
    x, y = load_samples("./features.train")
    y_ova_8 = ova(y, 8)

    c_list = [1e-5, 1e-3, 1e-1, 1e1, 1e3]
    e_in_list = []
    for c in c_list:
        arg_option = "-t 1 -d 2 -r 1 -g 1 -c " + str(c)
        model = svm_train(y_ova_8, x, arg_option)
        _, p_acc, _ = svm_predict(y_ova_8, x, model)
        e_in_list.append((100 - p_acc[0]) / 100)

    print(e_in_list)
    plt.scatter(np.log10(c_list), e_in_list)
    plt.show()


def sv_versus_c():
    x, y = load_samples("./features.train")
    y_ova_8 = ova(y, 8)

    c_list = [1e-5, 1e-3, 1e-1, 1e1, 1e3]
    sv_list = []
    for c in c_list:
        arg_option = "-t 1 -d 2 -r 1 -g 1 -c " + str(c)
        model = svm_train(y_ova_8, x, arg_option)
        nr_sv = model.get_nr_sv()
        sv_list.append(nr_sv)

    print(sv_list)
    plt.scatter(np.log10(c_list), sv_list)
    plt.show()


def sv_versus_gamma():
    x, y = load_samples("./features.train")
    y_ova_8 = ova(y, 8)

    g_list = list(range(-10, 10))
    sv_list = []
    for g in g_list:
        gamma = pow(10, g)
        arg_option = "-t 2 -c 1 -g " + str(gamma)
        model = svm_train(y_ova_8, x, arg_option)
        nr_sv = model.get_nr_sv()
        sv_list.append(nr_sv)

    print(sv_list)
    plt.scatter(g_list, sv_list)
    plt.show()


def rbf_kernel(x, gamma):
    ns = np.linalg.norm(x, axis=1) ** 2
    return np.exp(-gamma * (ns[:, None] + ns[None, :] - 2 * np.dot(x, x.T)))


def margin_versus_c():
    x, y = load_samples("./features.train")
    y_ova_0 = ova(y, 0)

    c_list = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
    margin_list = []
    for c in c_list:
        arg_option = "-q -t 2 -g 80 -c " + str(c)
        model = svm_train(y_ova_0, x, arg_option)
        sv_coef = np.squeeze(np.array(model.get_sv_coef()))
        sv = pd.DataFrame(model.get_SV()).values[:, 1:]
        sv_indices = np.add(model.get_sv_indices(), -1)
        coef_y = sv_coef * y[sv_indices]
        w_square = np.sum(rbf_kernel(sv, 80) * np.dot(coef_y, coef_y.T))
        w_norm = sqrt(w_square)
        margin_list.append(1 / w_norm)

    print("margin in Z space:")
    print(margin_list)
    plt.scatter(np.log10(c_list), margin_list)
    plt.show()


def margin_versus_gamma():
    x, y = load_samples("./features.train")
    y_ova_0 = ova(y, 0)

    g_list = list(range(-10, 10))
    margin_list = []
    for g in g_list:
        gamma = pow(10, g)
        arg_option = "-t 2 -c 1 -g " + str(gamma)
        model = svm_train(y_ova_0, x, arg_option)
        sv_coef = np.squeeze(np.array(model.get_sv_coef()))
        sv = pd.DataFrame(model.get_SV()).values[:, 1:]
        sv_indices = np.add(model.get_sv_indices(), -1)
        coef_y = sv_coef * y[sv_indices]
        w_square = np.sum(rbf_kernel(sv, gamma) * np.dot(coef_y, coef_y.T))
        w_norm = sqrt(w_square)
        margin_list.append(1 / w_norm)

    print(margin_list)
    plt.ylim(0, 1e-3)
    plt.scatter(g_list, margin_list)
    plt.show()


def e_out_versus_gamma():
    train_x, train_y = load_samples("./features.train")
    test_x, test_y = load_samples("./features.test")
    train_y_ova_0 = ova(train_y, 0)
    test_y_ova_0 = ova(test_y, 0)

    gamma_list = [1e0, 1e1, 1e2, 1e3, 1e4]
    e_out_list = []
    for g in gamma_list:
        arg_option = "-t 2 -c 0.1 -g " + str(g)
        model = svm_train(train_y_ova_0, train_x, arg_option)
        _, p_acc, _ = svm_predict(test_y_ova_0, test_x, model)
        e_out_list.append((100 - p_acc[0]) / 100)

    print(e_out_list)
    plt.scatter(np.log10(gamma_list), e_out_list)
    plt.show()


def e_val_versus_gamma():
    x, y = load_samples("./features.train")

    gamma_list = [1e-1, 1e0, 1e1, 1e2, 1e3]
    gamma_winner = np.zeros(5)
    for index in range(100):
        print("test", index)
        e_val_list = np.zeros(5)
        val_index = np.random.choice(np.arange(len(y)), 1000, replace=False)
        val_x = x[val_index]
        val_y = y[val_index]
        train_x = np.delete(x, val_index, axis=0)
        train_y = np.delete(y, val_index)
        val_y_ova_0 = ova(val_y, 0)
        train_y_ova_0 = ova(train_y, 0)

        for i, g in enumerate(gamma_list):
            arg_option = "-q -t 2 -c 0.1 -g " + str(g)
            model = svm_train(train_y_ova_0, train_x, arg_option)
            _, p_acc, _ = svm_predict(val_y_ova_0, val_x, model)
            e_val_list[i] = (100 - p_acc[0]) / 100

        gamma_winner[np.argmin(e_val_list)] += 1

    print(gamma_winner)
    plt.bar(np.log10(gamma_list), gamma_winner)
    plt.show()


if __name__ == '__main__':
    #weight_versus_c()
    #e_in_versus_c()
    #sv_versus_c()
    #sv_versus_gamma()
    #margin_versus_c()
    margin_versus_gamma()
    #e_out_versus_gamma()
    #e_val_versus_gamma()
