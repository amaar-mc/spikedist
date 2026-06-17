from collections.abc import Sequence
from math import exp, sqrt

from ._validate import check_train


def _kernel_sum(x: list[float], y: list[float], tau: float) -> float:
    """Naive O(n*m) sum of exp(-|xi - yj| / tau) over all spike pairs. Kept as the
    reference against which the fast markage path is tested."""
    total = 0.0
    for xi in x:
        for yj in y:
            total += exp(-abs(xi - yj) / tau)
    return total


def _self_sum(times: list[float], tau: float) -> float:
    """Saa = sum over all ordered pairs of exp(-|ti - tj| / tau) for one sorted train,
    in O(n) via the markage recursion. The diagonal contributes n; each off-diagonal
    pair is counted twice."""
    n = len(times)
    if n == 0:
        return 0.0
    total = float(n)
    markage = 0.0  # sum over earlier spikes of exp(-(t_i - t_j) / tau), carried forward
    for i in range(1, n):
        decay = exp(-(times[i] - times[i - 1]) / tau)
        markage = decay * (1.0 + markage)
        total += 2.0 * markage
    return total


def _cross_sum(a: list[float], b: list[float], tau: float) -> float:
    """Sab = sum over all pairs of exp(-|ai - bj| / tau) for two sorted trains, in
    O(n + m) via a single merge sweep. Carried sums are rescaled at each step, so the
    running values stay bounded and there is no overflow."""
    if not a or not b:
        return 0.0
    total = 0.0
    carry_a = 0.0  # sum over processed a-spikes of exp(-(t_now - a) / tau)
    carry_b = 0.0
    i = j = 0
    na, nb = len(a), len(b)
    prev: float | None = None
    while i < na or j < nb:
        take_a = j >= nb or (i < na and a[i] <= b[j])
        t = a[i] if take_a else b[j]
        if prev is not None:
            decay = exp(-(t - prev) / tau)
            carry_a *= decay
            carry_b *= decay
        if take_a:
            # Pair this a-spike with every earlier b-spike: exp(-(a - b) / tau).
            total += carry_b
            carry_a += 1.0
            i += 1
        else:
            # Pair this b-spike with every earlier a-spike: exp(-(b - a) / tau).
            total += carry_a
            carry_b += 1.0
            j += 1
        prev = t
    return total


def _distance_from_sums(saa: float, sbb: float, sab: float) -> float:
    squared = 0.5 * (saa + sbb - 2.0 * sab)
    # Guard against tiny negative values from floating-point cancellation.
    return sqrt(squared) if squared > 0.0 else 0.0


def van_rossum(a: Sequence[float], b: Sequence[float], *, tau: float) -> float:
    """Van Rossum distance between two spike trains.

    Each train is convolved with a causal exponential kernel ``exp(-t / tau)`` and the
    distance is the Euclidean distance between the filtered signals. Using the closed
    form of the kernel inner products, the squared distance is

        D^2 = 0.5 * (Saa + Sbb - 2 * Sab)

    where ``Sxy = sum over spike pairs exp(-|xi - yj| / tau)``. The sums are computed in
    O(n + m) time with the Houghton-Kreuz markage recursion rather than the naive O(n*m)
    double loop. With this normalization the distance between an empty train and a single
    spike is ``sqrt(0.5)``, and as ``tau`` grows large the distance approaches
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
    return _distance_from_sums(_self_sum(sa, tau), _self_sum(sb, tau), _cross_sum(sa, sb, tau))


def _van_rossum_naive(a: Sequence[float], b: Sequence[float], *, tau: float) -> float:
    """Reference O(n*m) implementation, used only to test the fast path."""
    if tau <= 0:
        raise ValueError(f"tau must be positive, received {tau}")
    sa = check_train("a", a)
    sb = check_train("b", b)
    return _distance_from_sums(
        _kernel_sum(sa, sa, tau), _kernel_sum(sb, sb, tau), _kernel_sum(sa, sb, tau)
    )
