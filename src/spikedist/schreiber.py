from collections.abc import Sequence
from math import exp, sqrt

from ._validate import check_train


def _gaussian_sum(x: list[float], y: list[float], sigma: float) -> float:
    """Sum of exp(-(xi - yj)^2 / (4 sigma^2)) over all spike pairs, proportional
    to the inner product of the two Gaussian-filtered trains."""
    denom = 4.0 * sigma * sigma
    total = 0.0
    for xi in x:
        for yj in y:
            d = xi - yj
            total += exp(-(d * d) / denom)
    return total


def schreiber(a: Sequence[float], b: Sequence[float], *, sigma: float) -> float:
    """Schreiber correlation-based similarity between two spike trains.

    Each train is convolved with a Gaussian of standard deviation ``sigma`` and
    the measure is the cosine similarity (normalized correlation) of the two
    filtered signals. The result lies in ``[0, 1]``: 1 for identical trains and
    approaching 0 as the trains share no near-coincident spikes.

    By convention two empty trains are perfectly similar (1.0) and a non-empty
    train compared with an empty one has similarity 0.0.

    Args:
        a: spike times of the first train.
        b: spike times of the second train.
        sigma: positive standard deviation of the Gaussian filter.

    Returns:
        The Schreiber similarity in ``[0, 1]``.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, received {sigma}")

    sa = check_train("a", a)
    sb = check_train("b", b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0

    saa = _gaussian_sum(sa, sa, sigma)
    sbb = _gaussian_sum(sb, sb, sigma)
    sab = _gaussian_sum(sa, sb, sigma)
    # Cauchy-Schwarz bounds this by 1; clamp tiny floating-point overshoot.
    return min(1.0, sab / sqrt(saa * sbb))
