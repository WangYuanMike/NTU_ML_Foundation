import numpy as np
import random
import sys
from itertools import cycle


class LinearClassifier:
    def __init__(self, train_file, test_file=None):
        self.train_data = np.loadtxt(train_file)
        self.train_x = np.ones(self.train_data.shape)
        self.train_x[:, :-1] = self.train_data[:, :-1]
        self.train_y = self.train_data[:, -1]

        if test_file is None:
            self.test_data = None
        else:
            self.test_data = np.loadtxt(test_file)

        self.weight = np.zeros(self.train_data.shape[1])

    def pla(self, index_pool, ita=1.0, verbose=False):
        counter = 0
        num_updates = 0
        while True:
            index = next(index_pool)
            y_hat = 1 if np.dot(self.train_x[index], self.weight) > 0 else -1
            if self.train_y[index] != y_hat:
                self.weight += ita * self.train_y[index] * self.train_x[index]
                num_updates += 1
                counter = 0
            else:
                counter += 1
                if counter == self.train_y.shape[0]:
                    break

        if verbose: print("num_updates =", num_updates)
        return num_updates

    def naive_pla(self):
        print()
        print("start naive pla...")
        index_pool = cycle(list(range(self.train_y.shape[0])))
        num_updates = self.pla(index_pool)
        print("num_updates =", num_updates)
        print("naive pla finished...")
        return num_updates

    def random_pla(self, ita=1.0, num_experiments=2000, verbose=False):
        print()
        print("start random pla...")
        avg_num_updates = 0
        for i in range(num_experiments):
            random.seed()
            index_list = list(range(self.train_y.shape[0]))
            random.shuffle(index_list)
            index_pool = cycle(index_list)
            if verbose: print("initialize weight...")
            self.init_weight()
            avg_num_updates += self.pla(index_pool, ita, verbose) / num_experiments

        print("ita =", ita)
        print("avg_num_updates =", avg_num_updates)
        print("random pla finished...")
        return avg_num_updates

    def pocket(self):
        return 0

    def init_weight(self):
        self.weight = np.zeros(self.train_data.shape[1])

    def head(self):
        print()
        print("train data x head:")
        print(self.train_x[:10])
        print("train data y head:")
        print(self.train_y[:10])
        if self.test_data is None:
            return
        print("test data head:")
        print(self.test_data[:10])

    def shape(self):
        print()
        print("train data x shape:", self.train_x.shape)
        print("train data y shape:", self.train_y.shape)
        if self.test_data is None: return
        print("test data shape:", self.test_data.shape)

    def show_weight(self):
        print()
        print("weight:", self.weight)
        print("weight shape:", self.weight.shape)


if sys.argv[1] == "hw1_15":
    TRAIN_FILE = "./hw1_15_train.dat"

if sys.argv[1] == "hw1_18":
    TRAIN_FILE = "./hw1_18_train.dat"

lc = LinearClassifier(TRAIN_FILE)
lc.head()
lc.shape()
lc.show_weight()

if sys.argv[2] == "naive_pla":
    lc.naive_pla()
    lc.show_weight()

if sys.argv[2] == "random_pla":
    lc.random_pla()
    lc.show_weight()

if sys.argv[2] == "random_pla_0.5":
    lc.random_pla(ita=0.5)
    lc.show_weight()


