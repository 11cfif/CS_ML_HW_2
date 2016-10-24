import numpy as np
import scipy.stats as sp_stat
from scipy import linalg
# from sklearn.linear_model import LinearRegression

MAX_P_VALUE = 0.03

ACTION_REMOVED = "removed"
ACTION_LOG = "log"
COUNT_ATTEMPTS = 1000


# ------------  Math  ------------
class LinearRegression:
    def __init__(self):
        self.betta = np.array
        self.intercept = 0

    def fit(self, X, y):
        self.betta = linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y[:, np.newaxis])
        return self

    def predict(self, X):
        return X.dot(self.betta) + self.intercept


def fit_linear_regression(X, y):
    lr = LinearRegression().fit(X, y - y.mean())
    lr.intercept = y.mean()
    return lr


# def fit_linear_regression(X, y):
#     lr = LinearRegression().fit(X, y)
#     lr.intercept_ = y.mean()
#     return lr


def rmse(y_real, y_predict):
    temp = 0
    for index in range(len(y_real)):
        temp += (y_real[index] - y_predict[index]) ** 2
    return np.sqrt(temp / len(y_real))


# ------------  CV  ------------
def divide_data(X, y, learn_fraction):
    learn_length = len(X) * learn_fraction
    indexes = range(len(X))
    learn_x = np.ones((learn_length, X.shape[1]))
    learn_y = np.ones(learn_length)
    for index in range(int(learn_length)):
        rand_index = indexes[np.random.random_integers(0, len(indexes) - 1)]
        learn_x[index] = X[rand_index].copy()
        learn_y[index] = y[rand_index].copy()
        indexes.remove(rand_index)
    test_length = len(X) - learn_length
    test_x = np.ones((test_length, X.shape[1]))
    test_y = np.ones(test_length)
    for index in range(int(test_length)):
        test_x[index] = X[indexes[index]].copy()
        test_y[index] = y[indexes[index]].copy()
    return learn_x, learn_y, test_x, test_y


# ------------  Tests  ------------
def update_factors(X, y):
    indexes = range(X.shape[1])
    good_factors = []
    temp = X.copy()
    for index in range(temp.shape[1]):
        print('iteration #%d' % index)
        rand_factor_index = indexes[np.random.random_integers(0, len(indexes) - 1)]
        temp, good = check_factor(temp, y, rand_factor_index)
        if good:
            good_factors.append(rand_factor_index)
        indexes.remove(rand_factor_index)
    return good_factors


def check_factor(X, y, factor_index):
    rmse_sample_old, rmse_sample_new = get_rmse_sample(X, y, factor_index)
    p_value = sp_stat.wilcoxon(rmse_sample_old, rmse_sample_new, zero_method="wilcox").pvalue
    print('factor_index = %d, p_value = %f' % (factor_index, p_value))
    if p_value > MAX_P_VALUE:
        return X, False
    return X, np.mean(rmse_sample_new) < np.mean(rmse_sample_old)


def get_rmse_sample(X, y, factor_index):
    rmse_sample_old = np.zeros(COUNT_ATTEMPTS)
    rmse_sample_new = np.zeros(COUNT_ATTEMPTS)
    for index in range(COUNT_ATTEMPTS):
        learn_x_old, learn_y, test_x_old, test_y = divide_data(X, y, 0.5)
        learn_x_new = np.delete(learn_x_old, factor_index, 1)
        test_x_new = np.delete(test_x_old, factor_index, 1)
        rmse_sample_old[index] = rmse(test_y, fit_linear_regression(learn_x_old, learn_y).predict(test_x_old).ravel())
        rmse_sample_new[index] = rmse(test_y, fit_linear_regression(learn_x_new, learn_y).predict(test_x_new).ravel())
    return rmse_sample_old, rmse_sample_new


# ------------  Util  ------------
def write_result(id, y):
    f = open('result.csv', 'w')  # opens the workfile file
    f.write('id,target\n')
    for i in range(len(y)):
        f.write(str(int(id[i])))
        f.write(',')
        f.write(str(y[i]))
        if i + 1 != len(y):
            f.write('\n')
    f.close()


def __main__():
    learn_path = "learn.csv"
    learn_data = np.loadtxt(learn_path, delimiter=',')
    test_path = "test.csv"
    test_data = np.loadtxt(test_path, delimiter=',')
    test_id = test_data[:, 0]
    test_x = test_data[:, 1:]
    learn_x = learn_data[:, 1:-1]
    learn_y = learn_data[:, -1]
    good_factors = update_factors(learn_x, learn_y)
    learn_good_x = learn_x[:, good_factors].copy()
    print(good_factors)
    print(rmse(learn_y, fit_linear_regression(test_x, learn_y).predict(test_x)))
    print(rmse(learn_y, fit_linear_regression(learn_good_x, learn_y).predict(learn_good_x)))
    print("------- test -------")
    test_good_x = test_x[:, good_factors].copy()

    y_predict = fit_linear_regression(learn_good_x, learn_y).predict(test_good_x).ravel()
    write_result(test_id, y_predict)

__main__()
