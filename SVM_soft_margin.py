import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cvxopt import solvers, matrix
from scipy import sign

import PLA_SVM as ps


class SVM:

    EPSILON = 1e-4

    def __init__(self, train_x, train_y, test_x, test_y, c=1, kernel=None, q=1, zeta=0, gamma=1):
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
        # Lagrange multipliers, training sample feature and class of support vectors
        self.sv_alpha = None
        self.sv_x = None
        self.sv_y = None
        # and these stuff of free support vectors
        self.free_sv_alpha = None
        self.free_sv_x = None
        self.free_sv_y = None

    @staticmethod
    def load_samples(sample_file):
        samples = np.loadtxt(sample_file)
        x = np.ones(samples.shape)
        x[:, 1:] = samples[:, 1:]  # x = [1, x0, x1]
        y = samples[:, 0]
        return x, y

    @staticmethod
    def plot_samples(x, y):
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(x[:,1], x[:,2], s=32, c=y, cmap=cm.get_cmap('Set1'), marker='*')
        plt.colorbar(sc)

    def fit(self):
        """
        - Use quadratic programming to fit the model
        - Kernel SVM fit could not provide explicit weight and bias for prediction
          Instead, it provides alpha, training sample feature and class of
          support vectors and free support vectors for prediction
        """
        def polynomial_kernel():
            """dot(NxD, DxN) = NxN"""
            return np.power(self.zeta + self.gamma * np.dot(self.train_x, self.train_x.T), self.q)

        def gaussian_kernel():
            """dot(NxD, DxN) = NxN"""
            ns = np.linalg.norm(self.train_x, 2) ** 2
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
        solvers.options['show_progress'] = False
        sol = solvers.qp(Q, p, G, h, A, b)

        alpha = np.squeeze(sol['x'])
        sv = np.squeeze(np.where(self.alpha > SVM.EPSILON))
        free_sv = np.squeeze(np.where((self.alpha > SVM.EPSILON) & (self.alpha / self.c < 0.99)))
        self.sv_alpha = alpha[sv]
        self.sv_x = self.train_x[sv]
        self.sv_y = self.train_y[sv]
        self.free_sv_alpha = alpha[free_sv]
        self.free_sv_x = self.train_x[free_sv]
        self.free_sv_y = self.train_y[free_sv]

    def predict(self, x_predict):
        """
        Predict the class for sample feature x_predict
        :param x_predict: feature vector of the predict sample (D x 1)
        :return: predict class and predict score
        """
        def polynomial_kernel(x):
            """dot(SV_Num x D, Dx1) = SV_Num x 1"""
            return np.power(self.zeta + self.gamma * np.dot(self.sv_x, x), self.q)

        def gaussian_kernel(x):
            """dot(SV_Num x D, Dx1) = SV_Num x 1"""
            return np.exp(-self.gamma * np.linalg.norm(self.sv_x - x, 2) ** 2)

        def kernel_sum(x):
            """dot(1 x SV_Num, SV_Num x 1) = 1"""
            if self.kernel is None or self.kernel == 'p':
                return np.dot(self.sv_alpha * self.sv_y, polynomial_kernel(x))
            elif self.kernel == 'g':
                return np.dot(self.sv_alpha * self.sv_y, gaussian_kernel(x))

        b = self.free_sv_y[0] - kernel_sum(self.free_sv_x[0])
        predict_score = kernel_sum(x_predict) + b
        return sign(predict_score), predict_score

    def draw_decision_boundary(self, x1_low=-1.0, x1_hi=1.0, x2_low=-1.0, x2_hi=1.0, offset = 1e-2):
        x1_list = np.arange(x1_low, x1_hi, offset)
        x2_list = np.arange(x2_low, x2_hi, offset)
        boundary = np.zeros((x1.shape[0], 2))

        for index, x1 in enumerate(x1_list):
            for x2 in x2_list:
                x = np.array([x1, x2])
                _, predict_score = self.predict(x)
                if abs(predict_score) < SVM.EPSILON:
                    boundary[index] = x
                    break

        plt.scatter(boundary[:, 0], boundary[:, 1], marker='.')


if __name__ == '__main__':

    def unit_test():
        N = 100
        target_w, target_b = ps.generate_target_weight_bias()
        x, y = ps.generate_samples(N, target_w, target_b)
        ps.print_sample(x, y)
        ps.wbline(target_w, target_b)

        pla_w, pla_b = ps.get_pla_weight_bias(x, y)
        ps.wbline(pla_w, pla_b, fmt=":")

        #TODO: load data, iniitialize SVM instance
        #fit()
        #draw_decision_boundary()

        plt.show()

        '''
        train_file = "./features.train"
        test_file = "./features.test"
        train_x, train_y = load_samples(train_file)
        test_x, test_y = load_samples(test_file)
        plot_samples(train_x, train_y)
        #plot_samples(test_x, test_y)
        plt.show()
        '''

    unit_test()