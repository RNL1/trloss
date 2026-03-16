"""Maximum Mean Discrepancy (MMD) estimators.

Used as a baseline distribution-matching criterion for model selection.
"""

import numpy as np
from sklearn.metrics import pairwise as kernels


def mmd_linear(X, Y):
    """Linear-time MMD estimator (Gretton et al., 2012).

    Parameters
    ----------
    X : array-like of shape (n, d)
        Source-domain representations.
    Y : array-like of shape (m, d)
        Target-domain representations.

    Returns
    -------
    mmd : float
        Squared MMD estimate.
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


def mmd_rbf(X, Y, gamma=1.0):
    """RBF-kernel MMD estimator.

    Parameters
    ----------
    X : array-like of shape (n, d)
        Source-domain representations.
    Y : array-like of shape (m, d)
        Target-domain representations.
    gamma : float, default=1.0
        RBF kernel bandwidth parameter.

    Returns
    -------
    mmd : float
        Squared MMD estimate.
    """
    XX = kernels.rbf_kernel(X, X, gamma)
    YY = kernels.rbf_kernel(Y, Y, gamma)
    XY = kernels.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()
