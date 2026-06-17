from bisect import bisect_left
from collections.abc import Sequence
from math import exp

from ._validate import check_train


def _nearest_distance(t: float, sorted_train: list[float]) -> float:
    """Absolute distance from t to the nearest spike in a non-empty sorted train."""
    i = bisect_left(sorted_train, t)
    best = float("inf")
    if i < len(sorted_train):
        best = sorted_train[i] - t
    if i > 0:
        best = min(best, t - sorted_train[i - 1])
    return best


def _directional(source: list[float], target: list[float], tau: float) -> float:
    """Mean over source spikes of exp(-nearest distance to target / tau).

    Both trains must be non-empty and the target sorted ascending."""
    total = 0.0
    for t in source:
        total += exp(-_nearest_distance(t, target) / tau)
    return total / len(source)


def hunter_milton(a: Sequence[float], b: Sequence[float], *, tau: float) -> float:
    """Hunter-Milton similarity between two spike trains.

    For each spike, the nearest spike in the other train is found and scored by
    ``exp(-dt / tau)``. The measure averages this score over both trains, giving
    a value in ``(0, 1]``: 1 for identical trains and decaying as spikes drift
    apart relative to ``tau``.

    By convention two empty trains are perfectly similar (1.0) and a non-empty
    train compared with an empty one has similarity 0.0.

    Args:
        a: spike times of the first train.
        b: spike times of the second train.
        tau: positive time constant.

    Returns:
        The Hunter-Milton similarity in ``(0, 1]``.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, received {tau}")

    sa = check_train("a", a)
    sb = check_train("b", b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0

    return 0.5 * (_directional(sa, sb, tau) + _directional(sb, sa, tau))
