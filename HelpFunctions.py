import sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def extend_mesh(x: np.ndarray) -> np.ndarray:
    """
    Function for extending mesh from center points to edge points

    :param x: Mesh to extend
    :type x: numpy.ndarray
    :return: The extended mesh
    :rtype: numpy.ndarray
    """
    x_delta = (np.diff(x)) / 2
    x_extended = x[:-1] + x_delta
    x_extended = np.insert(x_extended, 0, x[0] - x_delta[0])
    x_extended = np.append(x_extended, x[-1] + x_delta[-1])
    return x_extended


def progress(count: int, total: int, status='') -> None:
    """
    Progress bar for sys

    :param count: Current count
    :type count: int
    :param total: Total counts
    :type total: int
    :param status: Optional status string to display
    :type status: str
    :return: None
    """
    bar_len = 30
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %4.1f%s ...%s' % (bar, percents, '%', status))
    sys.stdout.flush()


def baseline_als(y, lam, p, niter=10):
    """
    Baseline fit according to:
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library

    :param y:
    :param lam:
    :param p:
    :param niter:
    :return:
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + np.multiply(lam, D.dot(D.transpose()))
        z = spsolve(Z, np.multiply(w, y))
        w = np.multiply(p, (y > z)) + np.multiply((1 - p), (y < z))
    return z
