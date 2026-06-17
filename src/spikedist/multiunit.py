from collections.abc import Mapping, Sequence
from math import sqrt

from ._validate import check_train
from .van_rossum import _cross_sum


def van_rossum_multiunit(
    a: Mapping[str, Sequence[float]],
    b: Mapping[str, Sequence[float]],
    *,
    tau: float,
    c: float,
) -> float:
    """Multi-unit van Rossum distance between two labeled populations of spike trains.

    Each argument maps a unit label to that unit's spike train; a unit absent from one
    side is treated as an empty train. The parameter ``c`` in ``[0, 1]`` sets how much
    spikes of different units interact: ``c = 0`` treats the units as independent, so the
    distance is the Euclidean combination of the per-unit van Rossum distances; ``c = 1``
    ignores the labels, so it equals the van Rossum distance of the pooled trains. With a
    single unit the result equals the ordinary van Rossum distance for any ``c``.

    This is the Houghton-Sen construction: the squared distance is ``0.5 * (Saa + Sbb -
    2 Sab)`` where each cross term is summed over unit pairs and weighted by 1 for the same
    unit and by ``c`` for different units.

    Args:
        a: labeled spike trains of the first population.
        b: labeled spike trains of the second population.
        tau: positive time constant of the exponential kernel.
        c: cross-unit interaction in [0, 1].

    Returns:
        The multi-unit van Rossum distance.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, received {tau}")
    if not 0.0 <= c <= 1.0:
        raise ValueError(f"c must be between 0 and 1, received {c}")

    units = sorted(set(a) | set(b))
    sa: dict[str, list[float]] = {}
    sb: dict[str, list[float]] = {}
    for unit in units:
        sa[unit] = check_train(f"a[{unit}]", a[unit]) if unit in a else []
        sb[unit] = check_train(f"b[{unit}]", b[unit]) if unit in b else []

    total = 0.0
    for u in units:
        for v in units:
            weight = 1.0 if u == v else c
            if weight == 0.0:
                continue
            cross = (
                _cross_sum(sa[u], sa[v], tau)
                + _cross_sum(sb[u], sb[v], tau)
                - 2.0 * _cross_sum(sa[u], sb[v], tau)
            )
            total += weight * cross

    squared = 0.5 * total
    return sqrt(squared) if squared > 0.0 else 0.0
