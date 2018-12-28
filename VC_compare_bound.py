from math import exp, sqrt, log
import matplotlib.pyplot as plt

# growth function
def mh(N, dvc):
    return pow(N, dvc)


def quadratic_root(a, b, c):
    d = sqrt(pow(b,2)-4*a*c)
    return (-b+d)/(2*a)


dvc = 50
delta = 0.05
N = 10000.0

original_vc_bound = sqrt(8*log(4*mh(2*N, dvc)/delta)/N)
rademacher_penalty_bound = sqrt(2*log(2*N*mh(N, dvc))/N) + sqrt(2*log(1/delta)/N) + 1/N
variant_vc_bound = sqrt(16*log(2*mh(N, dvc)/sqrt(delta))/N)

devroye_a = 2 * N - 4
devroye_b = -4
devroye_c = -log(4*mh(pow(N,2), dvc)/delta)     #TODO:overflow

pv_a = N
pv_b = -2
pv_c = -log(6*mh(2*N,dvc)/delta)

x_list = ["original vc", "Radem penaulty", "P&V", "Devroye", "variant vc"]
y_list = [original_vc_bound, rademacher_penalty_bound,
          quadratic_root(pv_a, pv_b, pv_c),
          quadratic_root(devroye_a, devroye_b, devroye_c),
          variant_vc_bound]

plt.scatter(x_list, y_list)
plt.show()