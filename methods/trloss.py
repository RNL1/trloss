"""
Transferability Loss (TR-LOSS) for Safe Model Selection under Domain Shift.

Implements the TR-LOSS criterion from:

    Nikzad-Langerodi, R. & Fonseca Diaz, V.
    "Transferability Loss for Safe Model Selection under Domain Shift"
    ICLR 2026 — Catch, Adapt, and Operate: Monitoring ML Models Under Drift Workshop

TR-LOSS measures representational consistency between a source hypothesis
and its adapted counterpart evaluated on target-domain inputs. It is
model-agnostic and works with any UDA method that produces paired
source/adapted predictions.

    Regression  (Eq. 10):  J_reg = MSE_S + τ_MSE       (minimize)
    Classification (Eq. 11):  J_cls = ACC_S + τ_ACC     (maximize)
"""

import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score


def transferability_loss(y_source_pred, y_adapted_pred, task="regression"):
    """Compute the transferability loss between paired predictions.

    Given target-domain inputs, the source hypothesis h_s and adapted
    hypothesis h_t each produce predictions.  TR-LOSS quantifies their
    disagreement:

        Regression:     τ_MSE = MSE(h_s(X_t), h_t(X_t))
        Classification: τ_ACC = ACC(h_s(X_t), h_t(X_t))  (agreement rate)

    Parameters
    ----------
    y_source_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Predictions of the *source* hypothesis on target-domain data.
    y_adapted_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Predictions of the *adapted* hypothesis on the same target-domain data.
    task : {"regression", "classification"}, default="regression"
        - ``"regression"``: returns MSE between the two prediction sets.
        - ``"classification"``: returns the agreement rate (accuracy) between
          the two prediction sets.

    Returns
    -------
    tau : float
        The transferability loss value.

    Examples
    --------
    >>> import numpy as np
    >>> from trloss import transferability_loss
    >>> # Regression
    >>> tau = transferability_loss([1.0, 2.0, 3.0], [1.1, 2.0, 2.9])
    >>> round(np.sqrt(tau), 4)  # τ_rMSE
    0.0577
    >>> # Classification
    >>> tau = transferability_loss([0, 1, 2], [0, 1, 1], task="classification")
    >>> round(tau, 4)
    0.6667
    """
    y_source_pred = np.asarray(y_source_pred)
    y_adapted_pred = np.asarray(y_adapted_pred)

    if task == "regression":
        return mean_squared_error(y_source_pred, y_adapted_pred)
    elif task == "classification":
        return accuracy_score(y_source_pred, y_adapted_pred)
    else:
        raise ValueError(
            f"Unknown task '{task}'. Choose 'regression' or 'classification'."
        )


def selection_criterion(source_perf, tau, task="regression"):
    """Compute the composite model-selection criterion (Eq. 10 / 11).

    Parameters
    ----------
    source_perf : float
        Source-domain validation performance.
        - Regression: MSE (or RMSE²) on source validation set.
        - Classification: accuracy on source validation set.
    tau : float
        Transferability loss returned by :func:`transferability_loss`.
    task : {"regression", "classification"}, default="regression"
        - ``"regression"``:  J = MSE_S + τ_MSE  → **minimize**
        - ``"classification"``:  J = ACC_S + τ_ACC  → **maximize**

    Returns
    -------
    criterion : float
        The composite score.
    """
    if task == "regression":
        return source_perf + tau
    elif task == "classification":
        return source_perf + tau
    else:
        raise ValueError(
            f"Unknown task '{task}'. Choose 'regression' or 'classification'."
        )
