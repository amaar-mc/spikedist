from collections.abc import Sequence

from ._validate import check_train


def victor_purpura(a: Sequence[float], b: Sequence[float], *, cost: float) -> float:
    """Victor-Purpura distance between two spike trains.

    The distance is the minimum total cost to turn train ``a`` into train ``b``
    using three operations: insert a spike (cost 1), delete a spike (cost 1), and
    shift a spike in time by ``dt`` (cost ``cost * abs(dt)``). ``cost`` is the
    parameter usually written ``q``; it sets the time scale. With ``cost == 0``
    the distance counts only the difference in spike count; as ``cost`` grows,
    shifting becomes expensive and the distance approaches 2 per unmatched spike.

    Spikes are treated as an unordered set of times and sorted internally.

    Args:
        a: spike times of the first train.
        b: spike times of the second train.
        cost: non-negative time-shift cost ``q``.

    Returns:
        The Victor-Purpura distance.
    """
    if cost < 0:
        raise ValueError(f"cost must be non-negative, received {cost}")

    sa = check_train("a", a)
    sb = check_train("b", b)
    n, m = len(sa), len(sb)
    if n == 0:
        return float(m)
    if m == 0:
        return float(n)

    # Dynamic program over prefixes; keep only the previous row (O(m) memory).
    # prev[j] is the cost of aligning the first i-1 spikes of a with the first j of b.
    prev: list[float] = [float(j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur: list[float] = [float(i)] + [0.0] * m
        ai = sa[i - 1]
        for j in range(1, m + 1):
            shift = prev[j - 1] + cost * abs(ai - sb[j - 1])
            cur[j] = min(prev[j] + 1.0, cur[j - 1] + 1.0, shift)
        prev = cur
    return prev[m]
