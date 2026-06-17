from collections.abc import Sequence
from math import exp, sqrt

from ._validate import check_train


def _kernel_sum(x: list[float], y: list[float], tau: float) -> float:
    """Sum of exp(-|xi - yj| / tau) over all spike pairs, the exponential-kernel
    inner product of the two filtered trains (up to the shared tau/2 factor)."""
    total = 0.0
    for xi in x:
        for yj in y:
            total += exp(-abs(xi - yj) / tau)
    return total


def van_rossum(a: Sequence[float], b: Sequence[float], *, tau: float) -> float:
    """Van Rossum distance between two spike trains.

    Each train is convolved with a causal exponential kernel ``exp(-t / tau)``
    and the distance is the Euclidean distance between the filtered signals.
    Using the closed form of the kernel inner products, the squared distance is

        D^2 = 0.5 * (Saa + Sbb - 2 * Sab)

    where ``Sxy = sum over spike pairs exp(-|xi - yj| / tau)``. With this
    normalization the distance between an empty train and a single spike is
    ``sqrt(0.5)``, and as ``tau`` grows large the distance approaches
    ``abs(len(a) - len(b)) / sqrt(2)``.

    Args:
        a: spike times of the first train.
        b: spike times of the second train.
        tau: positive time constant of the exponential kernel.

    Returns:
        The van Rossum distance.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, received {tau}")

    sa = check_train("a", a)
    sb = check_train("b", b)

    saa = _kernel_sum(sa, sa, tau)
    sbb = _kernel_sum(sb, sb, tau)
    sab = _kernel_sum(sa, sb, tau)
    squared = 0.5 * (saa + sbb - 2.0 * sab)
    # Guard against tiny negative values from floating-point cancellation.
    return sqrt(squared) if squared > 0.0 else 0.0
