"""Backward-compatibility shim.

The transferability loss criterion has moved to the top-level module
:mod:`trloss`.  This file is retained only for backward compatibility.

Usage (new):
    from trloss import transferability_loss

Usage (legacy, still works):
    from kdaPLS.metrics import transferability_loss
"""

from trloss import transferability_loss  # noqa: F401
