from math import exp

epsilon = 0.05
dvc = 10
N_list = [420000, 440000, 460000, 480000, 500000]

# It is hard to get a close solution of N, because the formula below is quite complex,
# therefore N has to be approximated in this way
for N in N_list:
    delta = 4*pow(2*N, dvc)*exp(-1/8*pow(epsilon, 2)*N)
    print("N = %d, delta = %.4f" % (N, delta))
