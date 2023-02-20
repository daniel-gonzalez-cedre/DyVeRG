import numpy as np
from sklearn import metrics
from scipy.stats import wasserstein_distance as wasserstein


def mmd_wasserstein(X: np.ndarray, Y: np.ndarray, kernel: str = 'gaussian', gamma: float = 1.0, sigma: float = 1.0) -> float:
    kernel = lambda x, y: np.exp(-wasserstein(x, y) / (2 * (sigma**2)))
    expectation = lambda P, Q: np.mean([kernel(p, q) for p in P for q in Q])
    max_mean_discrepancy = expectation(X, X) + expectation(Y, Y) - (2 * expectation(X, Y))
    return max_mean_discrepancy


# code below this line was taken from Jindong Wang's public transfer learning GitHub repository
# https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
# @misc{jindongwang-transferlearning,
#     howpublished = {\url{http://transferlearning.xyz}},
#     title = {Everything about Transfer Learning and Domain Adapation},
#     author = {Wang, Jindong and others}
# }

# Compute MMD (maximum mean discrepancy) using numpy and scikit-learn.
def mmd_rbf(X, Y, gamma=1.0) -> float:
    '''MMD using RBF (Gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Keyword Arguments:
            gamma {float} -- [kernel parameter] (default: {1.0})
        Returns:
            [scalar] -- [MMD value]
    '''
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_poly(X, Y, degree=2, gamma=1, coef0=0) -> float:
    '''MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Keyword Arguments:
            degree {int} -- [degree] (default: {2})
            gamma {int} -- [gamma] (default: {1})
            coef0 {int} -- [constant item] (default: {0})
        Returns:
            [scalar] -- [MMD value]
    '''
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_linear(X, Y) -> float:
    '''MMD using linear kernel (i.e., k(x,y) = <x,y>)
        Note that this is not the original linear MMD, only the reformulated and faster version.
        The original version is:
            def mmd_linear(X, Y):
                XX = np.dot(X, X.T)
                YY = np.dot(Y, Y.T)
                XY = np.dot(X, Y.T)
                return XX.mean() + YY.mean() - 2 * XY.mean()
        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Returns:
            [scalar] -- [MMD value]
    '''
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


if __name__ == '__main__':
    a = np.arange(1, 10).reshape(3, 3)
    b = [[7, 6, 5], [4, 3, 2], [1, 1, 8], [0, 2, 5]]
    b = np.array(b)
    print(a)
    print(b)
    print(mmd_linear(a, b))  # 6.0
    print(mmd_rbf(a, b))  # 0.5822
    print(mmd_poly(a, b))  # 2436.5
