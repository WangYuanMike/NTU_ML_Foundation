from scipy import exp
import numpy as np


def f(weight):
    u = weight[0]
    v = weight[1]
    return exp(u) + exp(2*v) + exp(u*v) + pow(u, 2) - 2*u*v + 2*pow(v, 2) - 3*u - 2*v


def gradient(weight):
    u = weight[0]
    v = weight[1]
    epsilon = 1e-4
    return np.array([(f(np.array([u+epsilon, v])) - f(np.array([u-epsilon, v]))) / (2 * epsilon),
                     (f(np.array([u, v+epsilon])) - f(np.array([u, v-epsilon]))) / (2 * epsilon)])


def gradient_descent(weight, eta=0.01, num_iter=5):
    print("Gradient descent:")
    counter = 0
    while counter < num_iter:
        weight -= eta * gradient(weight)
        print("round %d: u = %.3f, v = %.3f, error = %.3f" % (counter, weight[0], weight[1], f(weight)))
        counter += 1
    return weight


def hessian(weight):
    u = weight[0]
    v = weight[1]
    partial_uu = exp(u) + exp((u+1)*v) + 2
    partial_uv = (u+1) * exp((u+1)*v) - 2
    partial_vv = 4 * exp(2*v) + u * exp(u*(v+1)) + 4
    return np.matrix([[partial_uu, partial_uv], [partial_uv, partial_vv]])


def newton_direction(weight, eta=1, num_iter=5):
    print("Newton direction:")
    counter = 0
    while counter < num_iter:
        delta = np.squeeze(np.asarray(hessian(weight).I.dot(gradient(weight))))
        weight -= eta * delta
        print("round %d: u = %.3f, v = %.3f, error = %.3f" % (counter, weight[0], weight[1], f(weight)))
        counter += 1
    return weight


if __name__ == '__main__':
    zero_weight = np.array([0.0, 0.0])
    print("Initial weight:", zero_weight)
    gradient_descent(zero_weight)
    newton_direction(zero_weight)
