from sklearn.svm import SVC
from sklearn.model_selection import KFold
from SVM_soft_margin import load_samples, plot_samples, ova, ovo
import numpy as np


def get_error(target_y, predict_y):
    error = np.zeros(predict_y.shape)
    error[target_y != predict_y] = 1
    return error.sum() / len(error)


def ova_experiment(train_x, train_y, test_x, test_y):
    for i in range(10):
        train_y_ova = ova(train_y, i)
        test_y_ova = ova(test_y, i)

        #plot_samples(train_x, train_y_ova)

        svm = SVC(C=0.01, kernel='poly', gamma=1, coef0=1, degree=2, verbose=False)
        svm.fit(train_x, train_y_ova)

        train_y_predict = svm.predict(train_x)
        test_y_predict = svm.predict(test_x)
        E_in = get_error(train_y_ova, train_y_predict)
        E_out = get_error(test_y_ova, test_y_predict)
        print("%d versus all: C=0.01, kernel='poly', gamma=1, zeta=1, Q=2, E_in=%.3f, E_out=%.3f, sv_num=%d" %
              (i, E_in, E_out, sum(svm.n_support_)))


def ovo_experiment(train_x, train_y, test_x, test_y):
    C_list = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    Q_list = [2, 5]

    for Q in Q_list:
        for C in C_list:
            train_x_1v5, train_y_1v5 = ovo(train_x, train_y, 1, 5)
            test_x_1v5, test_y_1v5 = ovo(test_x, test_y, 1, 5)

            svm = SVC(C=C, kernel='poly', gamma=1, coef0=1, degree=Q, verbose=False)
            svm.fit(train_x_1v5, train_y_1v5)

            train_y_predict = svm.predict(train_x_1v5)
            test_y_predict = svm.predict(test_x_1v5)
            E_in = get_error(train_y_1v5, train_y_predict)
            E_out = get_error(test_y_1v5, test_y_predict)
            print("1 versus 5: C=%.2e, kernel=poly, gamma=1, zeta=1, Q=%d, E_in=%.3f, E_out=%.3f, sv_num=%d" %
                  (C, Q, E_in, E_out, sum(svm.n_support_)))


def ovo_cv(x, y, n_fold, C_list):
    E_cv = np.zeros(len(C_list))
    kf = KFold(n_splits=n_fold, shuffle=True)
    for i, C in enumerate(C_list):
        for train, val in kf.split(x):
            svm = SVC(C=C, kernel='poly', gamma=1, coef0=1, degree=2, verbose=False)
            svm.fit(x[train], y[train])
            val_y_predict = svm.predict(x[val])
            E_cv[i] += get_error(y[val], val_y_predict) / n_fold
    winner = np.argmin(E_cv)
    return winner, E_cv


def ovo_cv_experiment(x, y, N=5000):
    x_1v5, y_1v5 = ovo(x, y, 1, 5)
    C_list = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    C_wins = np.zeros(len(C_list))
    E_cv_avg = np.zeros(len(C_list))
    for i in range(N):
        winner, E_cv = ovo_cv(x_1v5, y_1v5, 10, C_list)
        C_wins[winner] += 1
        E_cv_avg += E_cv / N
        print("Run %d: winner=%.0e, " % (i, C_list[winner]), "E_cv =", E_cv)
    print("C_wins =", C_wins)
    print("E_cv_avg =", E_cv_avg)


def ovo_rbf_experiment(train_x, train_y, test_x, test_y):
    C_list = [1e-2, 1e-0, 1e2, 1e4, 1e6]

    for C in C_list:
        train_x_1v5, train_y_1v5 = ovo(train_x, train_y, 1, 5)
        test_x_1v5, test_y_1v5 = ovo(test_x, test_y, 1, 5)

        svm = SVC(C=C, gamma=1, verbose=False)
        svm.fit(train_x_1v5, train_y_1v5)

        train_y_predict = svm.predict(train_x_1v5)
        test_y_predict = svm.predict(test_x_1v5)
        E_in = get_error(train_y_1v5, train_y_predict)
        E_out = get_error(test_y_1v5, test_y_predict)
        print("1 versus 5: C=%.2e, kernel=rbf, E_in=%.3f, E_out=%.3f, sv_num=%d" %
              (C, E_in, E_out, sum(svm.n_support_)))


if __name__ == '__main__':
    train_file = "./features.train"
    test_file = "./features.test"
    train_x, train_y = load_samples(train_file)
    test_x, test_y = load_samples(test_file)

    #ova_experiment(train_x, train_y, test_x, test_y)
    #ovo_experiment(train_x, train_y, test_x, test_y)
    ovo_cv_experiment(train_x, train_y)
    #ovo_rbf_experiment(train_x, train_y, test_x, test_y)
