import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cvxopt import solvers, matrix
from scipy import sign

import PLA_SVM as ps


class SVM:

    def __init__(self, train_x, train_y, test_x, test_y, c=1.0, kernel=None, q=1, zeta=0, gamma=1):
        """
        SVM soft margin dual problem with kernel function
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
            """(N, D) x (D, N) = (N, N)"""
            ns = np.linalg.norm(self.train_x, axis=1) ** 2
            return np.exp(-self.gamma * (ns[:, None] + ns[None, :] - 2 * np.dot(self.train_x, self.train_x.T)))

        if self.kernel is None or self.kernel == 'p':
            k = polynomial_kernel()
        elif self.kernel == 'g':
            k = gaussian_kernel()

        epsilon = 1e-4
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
        self.sv = np.squeeze(np.where(alpha > epsilon))
        self.free_sv = np.squeeze(np.where((alpha > epsilon) & (alpha / self.c < 0.99)))
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

        b = 0.0
        free_sv_num = len(self.free_sv)
        for i in range(free_sv_num):
            b += (self.free_sv_y[i] - kernel_sum(self.free_sv_x[i])) / free_sv_num

        predict_score = kernel_sum(x_predict) + b
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


def ova_transform(y, i):
    y_ova = y.copy()
    y_ova[y_ova != i] = -1
    return y_ova


if __name__ == '__main__':
    #SVM.unit_test()

    train_file = "./features.train"
    test_file = "./features.test"
    train_x, train_y = load_samples(train_file)
    test_x, test_y = load_samples(test_file)
    #plot_samples(train_x, train_y)

    # one vs.all experiments
    c = 0.01
    k = 'p'
    q = 2
    z = 1
    for i in range(10):
        svm = SVM(train_x, ova_transform(train_y, i), test_x, ova_transform(test_y, i), c=c, kernel=k, q=q, zeta=z)
        svm.fit(verbose=False)
        print("c=%.2e, kernel=%s, q=%d ,zeta=%d | %d vs. all | E_in=%.3f, E_out=%.3f sv_num=%d" %
              (c, k, q, z, i, svm.e_in(), svm.e_out(), len(svm.sv)))
