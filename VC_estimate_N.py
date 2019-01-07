from scipy import optimize
from scipy import exp, power


def f(N):
    return 4*power(2*N, dvc)*exp(-1/8*power(epsilon, 2)*N) - delta


if __name__ == '__main__':
    epsilon = 0.05
    dvc = 10
    delta = 0.05
    root = optimize.newton(f, 420000)
    print("root = %.2f" % root)