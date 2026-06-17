"""NumPy-backed pairwise van Rossum distance matrix.

This module is an OPTIONAL fast path. It is only importable when NumPy is
installed. The public API surface is ``van_rossum_matrix``, which is
re-exported from the package __init__ when numpy is available via the
``spikedist[fast]`` extra.

The pure-Python ``van_rossum`` remains the reference implementation and
the default when NumPy is not installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

import numpy as np

from ._validate import check_train
from .van_rossum import _self_sum


def _cross_sum_numpy(
    a: np.ndarray,
    b: np.ndarray,
    tau: float,
) -> float:
    """Sum of exp(-|ai - bj| / tau) over all pairs, computed via broadcasting.

    Equivalent to ``_cross_sum`` in van_rossum.py but uses NumPy outer
    difference rather than the merge-sweep recursion. Returns the same scalar;
    used as the inner kernel in ``van_rossum_matrix``.

    Args:
        a: 1-D array of sorted spike times for the first train.
        b: 1-D array of sorted spike times for the second train.
        tau: positive time constant of the exponential kernel.

    Returns:
        The sum as a Python float.
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    diff = np.abs(a[:, np.newaxis] - b[np.newaxis, :])
    return float(np.sum(np.exp(-diff / tau)))


def van_rossum_matrix(
    trains: Sequence[Sequence[float]],
    *,
    tau: float,
) -> list[list[float]]:
    """Pairwise van Rossum distance matrix for a list of spike trains, using NumPy.

    Computes the full N x N matrix ``D[i][j] = van_rossum(trains[i], trains[j],
    tau=tau)`` for all ordered pairs. The result is identical to calling
    ``pairwise(trains, partial(van_rossum, tau=tau))`` but is faster for large N
    or large trains because the cross-kernel sums are computed with NumPy
    broadcasting rather than a pure-Python merge sweep.

    The diagonal is always 0.0. The matrix is symmetric to floating-point
    precision. Requires NumPy (install with ``pip install spikedist[fast]``).

    Args:
        trains: a sequence of spike trains, each a sequence of spike times.
        tau: positive time constant of the exponential kernel.

    Returns:
        A list of rows; entry ``[i][j]`` is the van Rossum distance.

    Raises:
        ValueError: if ``tau <= 0`` or any spike time is non-finite.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, received {tau}")

    n = len(trains)
    if n == 0:
        return []

    sorted_trains: list[list[float]] = [
        check_train(f"trains[{i}]", t) for i, t in enumerate(trains)
    ]
    arrays: list[np.ndarray] = [np.array(t, dtype=np.float64) for t in sorted_trains]

    # Per-train self-sums: Sii = _self_sum(sorted_trains[i], tau). Reuse the
    # pure-Python O(n) markage recursion; there are only N of these.
    self_sums: list[float] = [_self_sum(t, tau) for t in sorted_trains]

    # Build the upper triangle of cross-sums via NumPy broadcasting.
    cross: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        cross[i][i] = self_sums[i]
        for j in range(i + 1, n):
            s = _cross_sum_numpy(arrays[i], arrays[j], tau)
            cross[i][j] = s
            cross[j][i] = s

    result: list[list[float]] = []
    for i in range(n):
        row: list[float] = []
        for j in range(n):
            squared = 0.5 * (self_sums[i] + self_sums[j] - 2.0 * cross[i][j])
            dist = sqrt(squared) if squared > 0.0 else 0.0
            row.append(dist)
        result.append(row)
    return result
