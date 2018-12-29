from scipy import optimize, exp, sqrt, log, power
import matplotlib.pyplot as plt


dvc = 50
delta = 0.05
N = 10000.0


# growth function
def mh(N, dvc):
    return power(N, dvc)


def pv_bound(epsilon):
    return sqrt((2*epsilon + log(6*mh(2*N, dvc)/delta))/N) - epsilon


def devroye(epsilon):
    return sqrt((4*epsilon*(1+epsilon) + log(4*mh(N**2, dvc)/delta))/(2*N)) - epsilon


original_vc_bound = sqrt(8*log(4*mh(2*N, dvc)/delta)/N)
rademacher_penalty_bound = sqrt(2*log(2*N*mh(N, dvc))/N) + sqrt(2*log(1/delta)/N) + 1/N
variant_vc_bound = sqrt(16*log(2*mh(N, dvc)/sqrt(delta))/N)

x_list = ["original vc", "Radem penaulty", "P&V", "Devroye", "variant vc"]
y_list = [original_vc_bound, rademacher_penalty_bound,
          optimize.newton(pv_bound, 0), optimize.newton(devroye, 0), variant_vc_bound]

plt.scatter(x_list, y_list)
plt.show()