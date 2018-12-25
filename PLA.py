import numpy as np
import random
import sys
from itertools import cycle


class Perceptron:
    def __init__(self, train_file, test_file=None):
        self.train_data = np.loadtxt(train_file)
        self.train_x = np.ones(self.train_data.shape)
        self.train_x[:, :-1] = self.train_data[:, :-1]
        self.train_y = self.train_data[:, -1]

        if test_file is None:
            self.test_data = None
        else:
            self.test_data = np.loadtxt(test_file)
            self.test_x = np.ones(self.test_data.shape)
            self.test_x[:, :-1] = self.test_data[:, :-1]
            self.test_y = self.test_data[:, -1]

        self.weight = np.zeros(self.train_data.shape[1])
        self.pocket_weight = np.zeros(self.weight.shape)

    def pla(self, index_pool, eta=1.0, verbose=False):
        counter = 0
        num_updates = 0
        while True:
            index = next(index_pool)
            y_hat = 1 if np.dot(self.train_x[index], self.weight) > 0 else -1
            if self.train_y[index] != y_hat:
                self.weight += eta * self.train_y[index] * self.train_x[index]
                num_updates += 1
                counter = 0
            else:
                counter += 1
                if counter == len(self.train_y):
                    break

        if verbose: print("num_updates =", num_updates)
        return num_updates

    def naive_pla(self):
        print()
        print("start naive pla...")
        index_pool = cycle(list(range(len(self.train_y))))
        num_updates = self.pla(index_pool)
        print("num_updates =", num_updates)
        print("naive pla finished...")
        return num_updates

    def random_pla(self, eta=1.0, verbose=False):
        random.seed()
        index_list = list(range(len(self.train_y)))
        random.shuffle(index_list)
        index_pool = cycle(index_list)
        if verbose: print("initialize weight...")
        self.init_weight()
        return self.pla(index_pool, eta, verbose)

    def random_pla_experiments(self, eta=1.0, num_experiments=2000, verbose=False):
        print()
        print("start random pla experiments...")
        avg_num_updates = 0
        for i in range(num_experiments):
            avg_num_updates += self.random_pla(eta, verbose) / num_experiments
        print("eta =", eta)
        print("avg_num_updates =", avg_num_updates)
        print("random pla experiments finished...")
        return avg_num_updates

    def get_error_rate(self, is_pocket=False, is_test=False):
        if is_pocket:
            weight = self.pocket_weight
        else:
            weight = self.weight

        if is_test:
            set_x = self.test_x
            set_y = self.test_y
        else:
            set_x = self.train_x
            set_y = self.train_y

        set_y_hat = np.dot(set_x, weight)
        set_y_hat[set_y_hat > 0] = 1
        set_y_hat[set_y_hat <= 0] = -1
        errors = (set_y_hat != set_y).astype(int)
        return np.sum(errors) / len(set_y)

    def is_better_weight(self, verbose=False):
        pocket_error_rate = self.get_error_rate(is_pocket=True, is_test=False)
        new_error_rate = self.get_error_rate(is_pocket=False, is_test=False)
        if verbose:
            print("pocket weight:", self.pocket_weight)
            print("new weight:", self.weight)
            print("pocket error rate:", pocket_error_rate)
            print("new error rate:", new_error_rate)
        if pocket_error_rate > new_error_rate:
            return True
        else:
            return False

    def pocket(self, pocket_num_updates=50, eta=1.0, verbose=False, is_pocket=True):
        random.seed()
        if verbose:
            print()
            print("start pocket...")
            print("initialize weight...")
        self.init_weight()

        r = 0
        while r < pocket_num_updates:
            if verbose: print("round", r)
            index = random.randint(0, len(self.train_y)-1)
            y_hat = 1 if np.dot(self.train_x[index], self.weight) > 0 else -1
            if self.train_y[index] != y_hat:
                self.weight += eta * self.train_y[index] * self.train_x[index]
                r += 1
                if self.is_better_weight(verbose):
                    self.pocket_weight = self.weight.copy()

        test_error_rate = self.get_error_rate(is_pocket=is_pocket, is_test=True)
        if verbose: print("pocket test error rate:", test_error_rate)
        return test_error_rate

    def pocket_experiment(self, eta=1.0, pocket_num_updates=50, num_experiements=2000, verbose=False, is_pocket=True):
        print()
        print("start pocket experiments...")
        avg_test_error_rate = 0.0
        for i in range(num_experiements):
            if verbose:
                print()
                print("experiment", i)
            avg_test_error_rate += self.pocket(pocket_num_updates, eta, verbose, is_pocket) / num_experiements
        print("avg_test_error_rate =", avg_test_error_rate)
        print("pocket experiments finished...")
        return avg_test_error_rate

    def init_weight(self):
        self.weight = np.zeros(self.train_data.shape[1])
        self.pocket_weight = np.zeros(self.weight.shape)

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
        print("pocket weight:", self.pocket_weight)


if sys.argv[1] == "hw1_15":
    TRAIN_FILE = "./hw1_15_train.dat"
    TEST_FILE = None

if sys.argv[1] == "hw1_18":
    TRAIN_FILE = "./hw1_18_train.dat"
    TEST_FILE = "./hw1_18_test.dat"

p = Perceptron(TRAIN_FILE, TEST_FILE)
p.head()
p.shape()
p.show_weight()

if sys.argv[2] == "naive_pla":
    p.naive_pla()
    p.show_weight()

if sys.argv[2] == "random_pla":
    p.random_pla_experiments()
    p.show_weight()

if sys.argv[2] == "random_pla_0.5":
    p.random_pla_experiments(eta=0.5)
    p.show_weight()

if sys.argv[2] == "pocket_50":
    p.pocket_experiment(pocket_num_updates=50, num_experiements=2000, verbose=False, is_pocket=True)
    p.show_weight()

if sys.argv[2] == "w50":
    p.pocket_experiment(pocket_num_updates=50, num_experiements=2000, verbose=False, is_pocket=False)
    p.show_weight()

if sys.argv[2] == "pocket_100":
    p.pocket_experiment(pocket_num_updates=100, num_experiements=2000, verbose=False, is_pocket=True)
    p.show_weight()