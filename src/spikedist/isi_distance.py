from collections.abc import Sequence

from ._validate import check_train


def _initial_isi(spikes: list[float], t_start: float, t_end: float) -> tuple[float, int]:
    """Return (nu, index) where nu is the ISI active at t_start and index is the
    position in spikes (-1 if the first spike is after t_start, 0 if at/before).

    Edge convention (matches pyspike isi_distance_python):
    - If spikes[0] > t_start: the interval before the first spike is an
      auxiliary ISI.  For N > 1 it is max(spikes[0] - t_start, spikes[1] - spikes[0]);
      for N == 1 it is spikes[0] - t_start.  index = -1.
    - If spikes[0] <= t_start: the first ISI is the gap to the next spike.
      For N > 1 that is spikes[1] - spikes[0]; for N == 1 it is t_end - spikes[0].
      index = 0.

    For the empty-train case call this function with spikes = [t_start, t_end].
    """
    n = len(spikes)
    if spikes[0] > t_start:
        nu = (
            max(spikes[0] - t_start, spikes[1] - spikes[0]) if n > 1 else spikes[0] - t_start
        )
        return nu, -1
    nu = (spikes[1] - spikes[0]) if n > 1 else (t_end - spikes[0])
    return nu, 0


def _trailing_isi(spikes: list[float], t_end: float) -> float:
    """Return the auxiliary ISI used after the last spike in the train.

    Edge convention (matches pyspike):
    - For N > 1: max(t_end - spikes[-1], spikes[-1] - spikes[-2]).
    - For N == 1: t_end - spikes[-1].
    """
    if len(spikes) > 1:
        return max(t_end - spikes[-1], spikes[-1] - spikes[-2])
    return t_end - spikes[-1]


def _next_isi(spikes: list[float], idx: int, t_end: float) -> float:
    """ISI active immediately after advancing to spikes[idx].

    If idx is the last spike, return the trailing auxiliary ISI; otherwise
    return the gap to the following spike.
    """
    return spikes[idx + 1] - spikes[idx] if idx < len(spikes) - 1 else _trailing_isi(spikes, t_end)


def isi_distance(
    a: Sequence[float],
    b: Sequence[float],
    *,
    interval: Sequence[float],
) -> float:
    """ISI-distance between two spike trains over a finite time interval.

    The instantaneous inter-spike-interval (ISI) function of a train at time t
    is the length of the ISI that contains t.  When t falls before the first
    spike or after the last spike an auxiliary ISI is used: if the first spike
    occurs after the interval start, the auxiliary ISI before it is
    max(first_spike - t_start, second_ISI) (or first_spike - t_start for a
    single-spike train); the auxiliary ISI after the last spike is
    max(t_end - last_spike, last_ISI) (or t_end - last_spike for a single-spike
    train).  An empty train is treated as containing two auxiliary spikes at
    t_start and t_end, giving a constant ISI of t_end - t_start.

    The pointwise ISI dissimilarity is

        I(t) = |isi_a(t) - isi_b(t)| / max(isi_a(t), isi_b(t))

    and the ISI-distance is its time-average over the interval:

        D_I = (1 / (t_end - t_start)) * integral_{t_start}^{t_end} I(t) dt.

    The result lies in [0, 1]: 0 for identical trains and 1 for maximally
    dissimilar trains.  The edge convention matches pyspike (Kreuz et al. 2007)
    exactly.

    Definition: Kreuz T, Haas JS, Morelli A, Abarbanel HDI, Politi A (2007),
    "Measuring spike train synchrony," J Neurosci Methods 165:151-161.

    Args:
        a: spike times of the first train.
        b: spike times of the second train.
        interval: two-element sequence [t_start, t_end] defining the time
            interval over which the distance is averaged.

    Returns:
        The ISI-distance in [0, 1].

    Raises:
        ValueError: if interval[0] >= interval[1] or inputs contain
            non-finite values.
    """
    t_start = float(interval[0])
    t_end = float(interval[1])
    if t_start >= t_end:
        raise ValueError(
            f"interval start must be strictly less than end, received [{t_start}, {t_end}]"
        )

    sa = check_train("a", a)
    sb = check_train("b", b)

    # Deduplicate spike times, matching pyspike's use of np.unique in
    # reconcile_spike_trains. Duplicate times produce a zero-length ISI which
    # would cause division by zero in I(t).
    sa = sorted(set(sa))
    sb = sorted(set(sb))

    # Represent an empty train as auxiliary spikes at the interval edges.
    # This matches pyspike's get_spikes_non_empty which returns [t_start, t_end].
    if not sa:
        sa = [t_start, t_end]
    if not sb:
        sb = [t_start, t_end]

    n1 = len(sa)
    n2 = len(sb)

    nu1, index1 = _initial_isi(sa, t_start, t_end)
    nu2, index2 = _initial_isi(sb, t_start, t_end)

    # Running integral (area accumulation) and the previous event time.
    prev_t = t_start
    area = 0.0

    # The first segment value.
    denom = max(nu1, nu2)
    # denom is always > 0 when t_start < t_end (interval has positive length).
    seg_val = abs(nu1 - nu2) / denom

    # Merge the two spike trains, advancing by whichever next spike comes first,
    # and update the running ISI values. Stops when both trains are exhausted.
    total_steps = n1 + n2 - 2  # total remaining spikes to process
    steps = index1 + index2  # starts at -2, -1, or 0

    while steps < total_steps:
        next_a = index1 + 1 < n1
        next_b = index2 + 1 < n2

        if next_a and (not next_b or sa[index1 + 1] < sb[index2 + 1]):
            # Advance in train a.
            index1 += 1
            event_t = sa[index1]
            area += seg_val * (event_t - prev_t)
            prev_t = event_t
            nu1 = _next_isi(sa, index1, t_end)
        elif next_b and (not next_a or sb[index2 + 1] < sa[index1 + 1]):
            # Advance in train b.
            index2 += 1
            event_t = sb[index2]
            area += seg_val * (event_t - prev_t)
            prev_t = event_t
            nu2 = _next_isi(sb, index2, t_end)
        else:
            # Both next spikes coincide exactly (sa[index1+1] == sb[index2+1]).
            index1 += 1
            index2 += 1
            event_t = sa[index1]
            area += seg_val * (event_t - prev_t)
            prev_t = event_t
            nu1 = _next_isi(sa, index1, t_end)
            nu2 = _next_isi(sb, index2, t_end)

        steps = index1 + index2
        denom = max(nu1, nu2)
        seg_val = abs(nu1 - nu2) / denom

    # Add the final segment from prev_t to t_end.
    area += seg_val * (t_end - prev_t)

    return area / (t_end - t_start)
