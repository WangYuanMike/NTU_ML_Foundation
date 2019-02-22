"""
SVM soft margin dual problem with kernel function
- implemented with cvxopt quadratic programming library
- has some trouble in choosing epsilon to distinguish support vectors
- also the test result is not correct comparing with sklearn SVC and LIBSVM
- but it is a good reference for SVM implementation details
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cvxopt import solvers, matrix
from scipy import sign

import PLA_SVM as ps


class SVM:

    def __init__(self, train_x, train_y, test_x, test_y, c=1.0, kernel=None, q=1, zeta=0, gamma=1):
        """
        :param train_x:
        :param train_y:
        :param test_x:
        :param test_y:
        :param c: soft margin noise tolerance
        :param kernel: None: linear kernel, 'p': polynomial kernel, 'g': gaussian kernel
        :param q: polynomial kernel(a, b) = power(zeta + gamma * dot(a, b), q)
        :param zeta: see above
        :param gamma: gaussian kernel(a, b) = exp(-gamma * 2-norm(a - b) ** 2)
        """
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.c = c
        self.kernel = kernel
        self.q = q
        self.zeta = zeta
        self.gamma = gamma
        # Index, Lagrange multipliers, training sample feature and class of support vectors
        self.sv = None
        self.sv_alpha = None
        self.sv_x = None
        self.sv_y = None
        # and these stuff of free support vectors
        self.free_sv = None
        self.free_sv_alpha = None
        self.free_sv_x = None
        self.free_sv_y = None
        # weight is not available in kernel SVM
        self.b = None

    def fit(self, verbose=False):
        """
        - Use quadratic programming to fit the model
        - Kernel SVM fit could not provide explicit weight and bias for prediction
          Instead, it provides alpha, training sample feature and class of
          support vectors and free support vectors for prediction
        """
        def polynomial_kernel():
            """(N, D) x (D, N) = (N, N)"""
            return np.power(self.zeta + self.gamma * np.dot(self.train_x, self.train_x.T), self.q)

        def gaussian_kernel():
            """(N, (N)) + ((N), N) - (N, D) x (D, N) = (N, N)"""
            ns = np.linalg.norm(self.train_x, axis=1) ** 2
            return np.exp(-self.gamma * (ns[:, None] + ns[None, :] - 2 * np.dot(self.train_x, self.train_x.T)))

        if self.kernel is None or self.kernel == 'p':
            k = polynomial_kernel()
        elif self.kernel == 'g':
            k = gaussian_kernel()


        N = len(self.train_y)
        Q = matrix(np.outer(self.train_y, self.train_y) * k)
        p = matrix(-1.0, (N, 1))
        G_0 = np.identity(N) * -1
        G_c = np.identity(N)
        G_np = np.zeros((2 * N, N))
        G_np[:N, :] = G_0
        G_np[N:, :] = G_c
        G = matrix(G_np)
        h_np = np.zeros((2 * N, 1))
        h_np[N:] += self.c
        h = matrix(h_np)
        A = matrix(self.train_y[None, :])
        b = matrix(0.0)
        solvers.options['show_progress'] = verbose
        sol = solvers.qp(Q, p, G, h, A, b)

        alpha = np.squeeze(sol['x'])
        epsilon = 1e-4
        self.sv = np.squeeze(np.where(alpha > epsilon))
        self.free_sv = np.squeeze(np.where((alpha > epsilon) & (self.c - alpha > epsilon)))
        self.sv_alpha = alpha[self.sv]
        self.sv_x = self.train_x[self.sv]
        self.sv_y = self.train_y[self.sv]
        self.free_sv_alpha = alpha[self.free_sv]
        self.free_sv_x = self.train_x[self.free_sv]
        self.free_sv_y = self.train_y[self.free_sv]

    def predict(self, x_predict):
        """
        Predict the class for sample feature x_predict
        :param x_predict: feature vector of the predict sample (D,)
        :return: predict class and predict score
        """
        def polynomial_kernel(x):
            """(SV_Num, D) x (D,) = (SV_Num,)"""
            return np.power(self.zeta + self.gamma * np.dot(self.sv_x, x), self.q)

        def gaussian_kernel(x):
            """norm((SV_Num, D) - ((SV_Num), D), axis=1) = (SV_Num,)"""
            return np.exp(-self.gamma * np.linalg.norm(self.sv_x - x, axis=1) ** 2)

        def kernel_sum(x):
            """(SV_Num,) x (SV_Num,) = 1"""
            if self.kernel is None or self.kernel == 'p':
                return np.dot(self.sv_alpha * self.sv_y, polynomial_kernel(x))
            elif self.kernel == 'g':
                return np.dot(self.sv_alpha * self.sv_y, gaussian_kernel(x))

        if self.b is None:
            b = 0.0
            free_sv_num = len(self.free_sv)
            for i in range(free_sv_num):
                b += (self.free_sv_y[i] - kernel_sum(self.free_sv_x[i])) / free_sv_num
            self.b = b
        predict_score = kernel_sum(x_predict) + self.b
        return sign(predict_score), predict_score

    def draw_decision_boundary(self, x0_low=-1.0, x0_hi=1.0, x1_low=-1.0, x1_hi=1.0, M=200):
        x0_list = np.arange(x0_low, x0_hi, (x0_hi - x0_low) / M)
        x1_list = np.arange(x1_low, x1_hi, (x1_hi - x1_low) / M)
        boundary = np.zeros((x0_list.shape[0], 2))
        upper_bound = np.zeros(boundary.shape)
        lower_bound = np.zeros(boundary.shape)

        epsilon = 5e-2
        for index, x0 in enumerate(x0_list):
            for x1 in x1_list:
                x = np.array([x0, x1])
                _, predict_score = self.predict(x)
                if abs(predict_score) < epsilon:
                    boundary[index] = x
                if abs(predict_score - 1) < epsilon:
                    upper_bound[index] = x
                if abs(predict_score + 1) < epsilon:
                    lower_bound[index] = x

        plt.scatter(boundary[:, 0], boundary[:, 1], s=8, marker='.')
        plt.scatter(upper_bound[:, 0], upper_bound[:, 1], s=2, c='k', marker='.')
        plt.scatter(lower_bound[:, 0], lower_bound[:, 1], s=2, c='k', marker='.')

    def draw_support_vector(self):
        plt.scatter(self.sv_x[:, 0], self.sv_x[:, 1], s=256, edgecolor='k', facecolor='None', marker='s')
        plt.scatter(self.free_sv_x[:, 0], self.free_sv_x[:, 1], s=256, edgecolor='k', facecolor='None', marker='*')

    def get_error(self, x, y):
        y_predict = np.zeros(len(y))
        for i, feature in enumerate(x):
            y_predict[i], _ = self.predict(feature)
        error = np.zeros(y_predict.shape)
        error[y != y_predict] = 1
        return error.sum() / len(error)

    def e_in(self):
        return self.get_error(self.train_x, self.train_y)

    def e_out(self):
        return self.get_error(self.test_x, self.test_y)

    @staticmethod
    def unit_test():
        N = 100
        target_w, target_b = ps.generate_target_weight_bias()
        x, y = ps.generate_samples(N, target_w, target_b)
        ps.print_sample(x, y)
        ps.wbline(target_w, target_b)

        pla_w, pla_b = ps.get_pla_weight_bias(x, y)
        ps.wbline(pla_w, pla_b, fmt="--")

        svm = SVM(x, y, x, y, c=0.01, kernel='p', q=2, zeta=1)
        svm.fit()
        print("unit test E_in=%.3f" % (svm.e_in()))
        svm.draw_decision_boundary()
        svm.draw_support_vector()
        plt.show()


def load_samples(sample_file):
    samples = np.loadtxt(sample_file)
    x = samples[:, 1:]
    y = samples[:, 0]
    return x, y


def plot_samples(x, y):
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(x[:, 0], x[:, 1], s=32, c=y, cmap=cm.get_cmap('Set1'), marker='*')
    plt.colorbar(sc)
    plt.show()


def ova(y, i):
    y_ova = y.copy()
    y_ova[y_ova != i] = -1
    y_ova[y_ova == i] = 1
    return y_ova


def ovo(x, y, i, j):
    x_ovo = x[(y == i) | (y == j)].copy()
    y_ovo = y[(y == i) | (y == j)].copy()
    y_ovo[y_ovo == j] = -1
    y_ovo[y_ovo == i] = 1
    return x_ovo, y_ovo


def main():
    train_file = "./features.train"
    test_file = "./features.test"
    train_x, train_y = load_samples(train_file)
    test_x, test_y = load_samples(test_file)
    # plot_samples(train_x, train_y)

    ova_p_kernel = True
    ovo_p_kernel = False
    ovo_g_kernel = False

    # one versus all / polynomial kernel
    if ova_p_kernel:
        c = 0.01
        k = 'p'
        q = 2
        z = 1
        for i in [2, 4, 6, 8]:
            svm = SVM(train_x, ova(train_y, i), test_x, ova(test_y, i), c=c, kernel=k, q=q, zeta=z)
            svm.fit(verbose=True)
            print("%d versus all: c=%.2e, kernel=%s, q=%d ,zeta=%d, E_in=%.3f, E_out=%.3f, sv_num=%d, free_sv_num=%d" %
                  (i, c, k, q, z, svm.e_in(), svm.e_out(), len(svm.sv), len(svm.free_sv)))

    # one versus one / polynomial kernel
    if ovo_p_kernel:
        k = 'p'
        q = 2
        z = 1
        c_list = [1e-4, 1e-3, 1e-2, 1e-1, 1]
        q_list = [2, 5]
        for q in q_list:
            for c in c_list:
                train_x_ovo, train_y_ovo = ovo(train_x, train_y, 1, 5)
                test_x_ovo, test_y_ovo = ovo(test_x, test_y, 1, 5)
                svm = SVM(train_x_ovo, train_y_ovo, test_x_ovo, test_y_ovo, c=c, kernel=k, q=q, zeta=z)
                svm.fit(verbose=False)
                print("1 versus 5: c=%.2e, kernel=%s, q=%d ,zeta=%d, E_in=%.3f, E_out=%.3f, sv_num=%d, free_sv_num=%d" %
                      (c, k, q, z, svm.e_in(), svm.e_out(), len(svm.sv), len(svm.free_sv)))

    # one versus one / gaussian kernel
    if ovo_g_kernel:
        k = 'g'
        g = 1
        c_list = [1e-2, 1, 1e2, 1e4, 1e6]
        for c in c_list:
            train_x_ovo, train_y_ovo = ovo(train_x, train_y, 1, 5)
            test_x_ovo, test_y_ovo = ovo(test_x, test_y, 1, 5)
            svm = SVM(train_x_ovo, train_y_ovo, test_x_ovo, test_y_ovo, c=c, kernel=k, gamma=g)
            svm.fit(verbose=False)
            print("1 versus 5: c=%.2e, kernel=%s, gamma=%d, E_in=%.3f, E_out=%.3f, sv_num=%d, free_sv_num=%d" %
                  (c, k, g, svm.e_in(), svm.e_out(), len(svm.sv), len(svm.free_sv)))


if __name__ == '__main__':
    #SVM.unit_test()

    main()