from collections.abc import Callable, Sequence

Metric = Callable[[Sequence[float], Sequence[float]], float]


def pairwise(trains: Sequence[Sequence[float]], metric: Metric) -> list[list[float]]:
    """Compute the full matrix ``metric(trains[i], trains[j])`` for all i, j.

    ``metric`` is any callable taking two spike trains and returning a float.
    Parameterize a metric with a lambda or ``functools.partial``, for example
    ``lambda a, b: van_rossum(a, b, tau=0.01)``. All n*n pairs are evaluated, so
    no symmetry is assumed and the cost is O(n^2) metric calls.

    Args:
        trains: a sequence of spike trains.
        metric: a callable comparing two spike trains.

    Returns:
        A list of rows; entry ``[i][j]`` is ``metric(trains[i], trains[j])``.
    """
    return [[metric(a, b) for b in trains] for a in trains]
