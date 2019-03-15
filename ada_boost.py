import numpy as np
import matplotlib.pyplot as plt
from scipy import sqrt, log, sign
import decision_stump as ds
import kernel_ridge_regression as krr


class AdaBoostStump:
    def __init__(self):
        self.alpha = []
        self.theta = []
        self.s = []
        self.i = []
        self.T = 0
        self.U = []
        self.epsilon = []

    def fit(self, x, y, T):
        N = len(y)
        u = np.ones(N) / N
        for t in range(T):
            epsilon, theta, s, i, correct, incorrect = ds.multi_dim_decision_stump(x, y, u)

            # store g(t)
            self.theta.append(theta)
            self.s.append(s)
            self.i.append(i)

            # store epsilon(t) and U(t) = sum(u(t))
            self.epsilon.append(epsilon)
            self.U.append(sum(u))

            # update u(t+1) with u(t) and diamond
            diamond = sqrt((1 - epsilon) / epsilon)
            u[incorrect] *= diamond
            u[correct] /= diamond

            # compute and store alpha(t)
            alpha = log(diamond)
            self.alpha.append(alpha)

        self.T = T

    def predict(self, x, y, T=None):
        score = 0
        if T is None:
            T = self.T
        for t in range(T):
            score += self.alpha[t] * self.h(x, t)
        predict_y = sign(score)
        error = krr.get_error(y, predict_y)
        return predict_y, error

    def predict_g(self, x, y, t):
        predict_y = self.h(x, t)
        error = krr.get_error(y, predict_y)
        return predict_y, error

    def h(self, x, t):
        return self.s[t] * sign(x[:, self.i[t]] - self.theta[t])

    def t_versus_g(self, x, y):
        error = []
        for t in range(self.T):
            _, error_t = self.predict_g(x, y, t)
            error.append(error_t)
        print(error)
        print(self.alpha)
        plt.plot(error)
        plt.show()

    def t_versus_G(self, x, y):
        error = []
        for t in range(1, self.T+1):
            _, error_t = self.predict(x, y, t)
            error.append(error_t)
        print(error)
        print(self.alpha)
        plt.plot(error)
        plt.show()

    def t_versus_u(self):
        print(self.U)
        plt.plot(self.U)
        plt.show()

    def t_versus_epsilon(self):
        print(self.epsilon)
        plt.plot(self.epsilon)
        plt.show()


def main():
    train_x, train_y = krr.load_samples("./hw3_train.dat")
    test_x, test_y = krr.load_samples("./hw3_test.dat")
    model = AdaBoostStump()
    model.fit(train_x, train_y, T=300)

    #model.t_versus_g(train_x, train_y)
    #model.t_versus_G(train_x, train_y)
    #model.t_versus_u()
    #model.t_versus_epsilon()
    #model.t_versus_g(test_x, test_y)
    model.t_versus_G(test_x, test_y)


if __name__ == '__main__':
    main()
