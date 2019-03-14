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

    def fit(self, x, y, T=300):
        N = len(y)
        u = np.ones(N) / N
        for t in range(T):
            e_in, theta, s, i, correct, incorrect = ds.multi_dim_decision_stump(x, y, u)

            # store g(t)
            self.theta.append(theta)
            self.s.append(s)
            self.i.append(i)

            # update u(t+1) with u(t) and diamond
            diamond = sqrt((1 - e_in) / e_in)
            u[incorrect] *= diamond
            u[correct] /= diamond

            # compute and store alpha(t)
            alpha = log(diamond)
            self.alpha.append(alpha)

        self.T = T

    def predict(self, x, y):
        score = 0
        for t in range(self.T):
            score += self.alpha[t] * self.h(x, t)
        predict_y = sign(score)
        error = krr.get_error(y, predict_y)
        return predict_y, error

    def predict_gt(self, x, y, t):
        predict_y = sign(self.alpha[t] * self.h(x, t))
        error = krr.get_error(y, predict_y)
        return predict_y, error

    def h(self, x, t):
        return self.s[t] * sign(x[:, self.i[t]] - self.theta[t])


def t_versus_gt():
    train_x, train_y = krr.load_samples("./hw3_train.dat")
    model = AdaBoostStump()
    model.fit(train_x, train_y, T=300)
    e_in = []
    for t in range(model.T):
        _, e_in_t = model.predict_gt(train_x, train_y, t)
        e_in.append(e_in_t)
    print(e_in)
    print(list(range(model.T)))
    plt.scatter(list(range(model.T)), e_in)
    plt.show()


if __name__ == '__main__':
    t_versus_gt()
