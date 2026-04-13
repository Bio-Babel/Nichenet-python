"""Utility functions for NicheNet data transformations.

Provides scaling, normalization, and helper functions ported from the
R ``nichenetr`` package with exact algorithmic parity.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

__all__ = [
    "scaling_zscore",
    "scaling_modified_zscore",
    "mapper",
    "scale_quantile",
    "scale_quantile_adapted",
]


def scaling_zscore(x: np.ndarray) -> np.ndarray:
    """Normalize values by the z-score method.

    Replicates the R ``scaling_zscore`` function.  When the standard
    deviation is zero the mean is still subtracted but no division occurs.
    A single-element input always returns ``0.0``.

    Parameters
    ----------
    x : np.ndarray
        Numeric array of values.

    Returns
    -------
    np.ndarray
        Z-score normalized values (same shape as *x*).
    """
    x = np.asarray(x, dtype=np.float64)

    if x.size == 1:
        return np.array(0.0)

    mean = np.nanmean(x)
    std = np.nanstd(x, ddof=1)  # R sd() uses n-1 denominator

    if std > 0:
        return (x - mean) / std
    else:
        return x - mean


def scaling_modified_zscore(x: np.ndarray) -> np.ndarray:
    """Normalize values by the modified z-score method.

    Uses median and median absolute deviation (MAD) instead of mean and
    standard deviation.  Replicates the R ``scaling_modified_zscore``
    function, including the 0.6745 consistency constant.

    Parameters
    ----------
    x : np.ndarray
        Numeric array of values.

    Returns
    -------
    np.ndarray
        Modified z-score normalized values (same shape as *x*).
    """
    x = np.asarray(x, dtype=np.float64)

    median = np.nanmedian(x)

    # R mad() default: median(abs(x - median(x))) * 1.4826
    mad = np.nanmedian(np.abs(x - median)) * 1.4826

    if mad != 0:
        return 0.6745 * (x - median) / mad
    else:
        return 0.6745 * (x - median)


def mapper(
    df: pd.DataFrame,
    value_col: str,
    name_col: str,
) -> Dict[str, object]:
    """Create a dictionary mapping from two DataFrame columns.

    Equivalent to the R expression
    ``setNames(df[[value_col]], df[[name_col]])``.

    Parameters
    ----------
    df : pandas.DataFrame
        Source data frame.
    value_col : str
        Column whose values become the dictionary *values*.
    name_col : str
        Column whose values become the dictionary *keys*.

    Returns
    -------
    dict
        Mapping from *name_col* entries to *value_col* entries.
    """
    return dict(zip(df[name_col], df[value_col]))


def scale_quantile(
    x: np.ndarray,
    outlier_cutoff: float = 0.05,
) -> np.ndarray:
    """Cut off outer quantiles and rescale to a [0, 1] range.

    Ported from the ``scale_quantile`` function in R ``nichenetr``
    (originally from ``dynutils``).  For each column (or the single
    vector) the lower and upper quantiles defined by *outlier_cutoff*
    are computed; values are linearly rescaled so that the lower
    quantile maps to 0 and the upper quantile maps to 1, then clamped
    to [0, 1].

    Parameters
    ----------
    x : np.ndarray
        Numeric vector or 2-D array.
    outlier_cutoff : float, optional
        Quantile fraction used at both tails (default 0.05).

    Returns
    -------
    np.ndarray
        Scaled values in [0, 1], same shape as *x*.
    """
    x = np.asarray(x, dtype=np.float64)
    is_1d = x.ndim < 2

    if is_1d:
        x = x.reshape(-1, 1)

    # Compute quantiles per column (ignoring NaNs)
    q_low = np.nanquantile(x, outlier_cutoff, axis=0)
    q_high = np.nanquantile(x, 1.0 - outlier_cutoff, axis=0)

    divisor = q_high - q_low
    divisor[divisor == 0] = 1.0

    # Linear rescale: shift so q_low -> 0, then scale so q_high -> 1
    y = (x - q_low) / divisor

    # Clamp to [0, 1]
    y = np.clip(y, 0.0, 1.0)

    if is_1d:
        y = y.ravel()

    return y


def scale_quantile_adapted(
    x: np.ndarray,
    outlier_cutoff: float = 0.0,
) -> np.ndarray:
    """Quantile scaling with an added pseudovalue.

    Calls :func:`scale_quantile` and then adds a pseudovalue of 0.001
    so that the lowest-ranked entry is never exactly zero.

    Parameters
    ----------
    x : np.ndarray
        Numeric vector or 2-D array.
    outlier_cutoff : float, optional
        Quantile fraction used at both tails (default 0).

    Returns
    -------
    np.ndarray
        Scaled values in [0.001, 1.001], same shape as *x*.
    """
    return scale_quantile(x, outlier_cutoff=outlier_cutoff) + 0.001
