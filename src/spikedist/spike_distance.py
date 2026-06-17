from collections.abc import Sequence

from ._validate import check_train


def _aux_boundaries(spikes: list[float], t_start: float, t_end: float) -> tuple[float, float]:
    """Return the auxiliary spike positions used for get_min_dist lookups.

    For a train with N > 1 spikes, extrapolate one ISI beyond each edge:
        aux_before = min(t_start, spikes[0] - (spikes[1] - spikes[0]))
        aux_after  = max(t_end,   spikes[-1] + (spikes[-1] - spikes[-2]))
    For N == 1, the auxiliary positions collapse to the interval edges.

    This matches pyspike's t_aux1/t_aux2 computation in spike_distance_python.
    """
    n = len(spikes)
    if n > 1:
        aux_before = min(t_start, spikes[0] - (spikes[1] - spikes[0]))
        aux_after = max(t_end, spikes[-1] + (spikes[-1] - spikes[-2]))
    else:
        aux_before = t_start
        aux_after = t_end
    return aux_before, aux_after


def _get_min_dist(
    spike_time: float,
    spike_train: list[float],
    start_index: int,
    aux_before: float,
    aux_after: float,
) -> float:
    """Return the minimum distance from spike_time to:
        - aux_before (the earlier boundary anchor),
        - all spike_train[i] for i >= start_index (early-exit when distance grows),
        - aux_after (the later boundary anchor).

    The early-exit is valid because spike_train is sorted ascending, so once
    distance starts increasing we have passed the nearest element.

    This matches pyspike's get_min_dist exactly.
    """
    d = abs(spike_time - aux_before)
    i = max(start_index, 0)
    while i < len(spike_train):
        d_temp = abs(spike_time - spike_train[i])
        if d_temp > d:
            return d
        d = d_temp
        i += 1
    d_temp = abs(aux_after - spike_time)
    return d if d_temp > d else d_temp


def _trailing_isi_spike(spikes: list[float], t_end: float) -> float:
    """Auxiliary ISI after the last spike, matching pyspike's edge convention:
        N > 1: max(t_end - spikes[-1], spikes[-1] - spikes[-2])
        N == 1: t_end - spikes[-1]
    """
    if len(spikes) > 1:
        return max(t_end - spikes[-1], spikes[-1] - spikes[-2])
    return t_end - spikes[-1]


def _dist_at_t(isi1: float, isi2: float, s1: float, s2: float) -> float:
    """Instantaneous SPIKE-distance at a point where the ISI contexts are isi1,
    isi2 and the local spike-time distances are s1, s2.

    Formula (MRTS=0, RI=False from pyspike):
        mean_isi = 0.5 * (isi1 + isi2)
        D = 0.5 * (s1 * isi2 + s2 * isi1) / (mean_isi * mean_isi)

    Returns 0.0 when mean_isi == 0 (degenerate case; guarded against division
    by zero).
    """
    mean_isi = 0.5 * (isi1 + isi2)
    if mean_isi == 0.0:
        return 0.0
    return 0.5 * (s1 * isi2 + s2 * isi1) / (mean_isi * mean_isi)


def spike_distance(
    a: Sequence[float],
    b: Sequence[float],
    *,
    interval: Sequence[float],
) -> float:
    """SPIKE-distance between two spike trains over a finite time interval.

    The SPIKE-distance measures the dissimilarity between two spike trains by
    tracking, at each time t, how far the nearest preceding and following spikes
    in each train are from the spikes in the other train, weighted by the local
    ISI context. The resulting time-resolved profile is piecewise-linear and its
    time-average is the scalar SPIKE-distance.

    At each time t that lies in an ISI (t_p, t_f) of train 1 the local spike-time
    distance is interpolated linearly between the distances dt_p (from the
    preceding spike to the nearest spike in train 2) and dt_f (from the following
    spike to the nearest spike in train 2):

        s_1(t) = (dt_p * (t_f - t) + dt_f * (t - t_p)) / isi_1

    and symmetrically for train 2. The instantaneous SPIKE-distance is then

        S(t) = 0.5 * (s_1(t) * isi_2 + s_2(t) * isi_1) / mean_isi^2

    where mean_isi = 0.5 * (isi_1 + isi_2). The scalar SPIKE-distance is the
    time-average of S(t) over [t_start, t_end]:

        D_S = (1 / (t_end - t_start)) * integral_{t_start}^{t_end} S(t) dt.

    Edge convention (matches pyspike exactly):
    - For each train, auxiliary positions are extrapolated one ISI beyond the
      first and last spike: aux_before = min(t_start, s_1 - (s_2 - s_1)) and
      aux_after = max(t_end, s_N + (s_N - s_{N-1})). For a single-spike train,
      aux_before = t_start and aux_after = t_end.
    - An empty train is replaced by auxiliary spikes at [t_start, t_end],
      matching pyspike's get_spikes_non_empty.
    - The leading ISI (for a train whose first spike is after t_start) is
      max(s_1 - t_start, s_2 - s_1) for N > 1, and s_1 - t_start for N == 1.
    - The trailing ISI is max(t_end - s_N, s_N - s_{N-1}) for N > 1, and
      t_end - s_N for N == 1.

    The profile is piecewise-linear between consecutive event times in the merged
    event sequence; the integral is the trapezoidal sum over each segment. The
    result lies in [0, 1]: 0 for identical trains.

    Reference: Kreuz T, Chicharro D, Houghton C, Andrzejak RG, Mormann F (2013),
    "Monitoring spike train synchrony," J Neurophysiol 109:1457-1472.

    Args:
        a: spike times of the first train.
        b: spike times of the second train.
        interval: two-element sequence [t_start, t_end] defining the time
            interval over which the distance is averaged.

    Returns:
        The SPIKE-distance in [0, 1].

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

    # Deduplicate spike times, matching pyspike's np.unique in reconcile_spike_trains.
    sa = sorted(set(sa))
    sb = sorted(set(sb))

    # Empty train -> auxiliary spikes at the interval edges, matching pyspike's
    # get_spikes_non_empty which returns [t_start, t_end].
    if not sa:
        sa = [t_start, t_end]
    if not sb:
        sb = [t_start, t_end]

    n1 = len(sa)
    n2 = len(sb)

    # Auxiliary boundary positions for each train (used by _get_min_dist).
    aux1_before, aux1_after = _aux_boundaries(sa, t_start, t_end)
    aux2_before, aux2_after = _aux_boundaries(sb, t_start, t_end)

    # --- Initialize train 1 ---
    # t_p1: position of the last "visited" spike (or the leading auxiliary position).
    # t_f1: position of the next upcoming spike (or the trailing auxiliary position).
    # dt_p1: distance from t_p1 to the nearest spike in train 2 (via _get_min_dist).
    # dt_f1: distance from t_f1 to the nearest spike in train 2.
    # isi1: length of the current ISI in train 1.
    # s1: interpolated spike-time distance for train 1 at the segment start.
    if sa[0] > t_start:
        t_f1 = sa[0]
        dt_f1 = _get_min_dist(t_f1, sb, 0, aux2_before, aux2_after)
        dt_p1 = dt_f1
        isi1 = max(t_f1 - t_start, sa[1] - sa[0]) if n1 > 1 else t_f1 - t_start
        s1 = dt_p1
        index1 = -1
        t_p1 = aux1_before
    else:
        t_f1 = sa[1] if n1 > 1 else t_end
        t_p1 = t_start
        dt_p1 = _get_min_dist(t_p1, sb, 0, aux2_before, aux2_after)
        dt_f1 = _get_min_dist(t_f1, sb, 0, aux2_before, aux2_after)
        isi1 = t_f1 - sa[0]
        s1 = dt_p1
        index1 = 0

    # --- Initialize train 2 ---
    if sb[0] > t_start:
        t_f2 = sb[0]
        dt_f2 = _get_min_dist(t_f2, sa, 0, aux1_before, aux1_after)
        dt_p2 = dt_f2
        isi2 = max(t_f2 - t_start, sb[1] - sb[0]) if n2 > 1 else t_f2 - t_start
        s2 = dt_p2
        index2 = -1
        t_p2 = aux2_before
    else:
        t_f2 = sb[1] if n2 > 1 else t_end
        t_p2 = t_start
        dt_p2 = _get_min_dist(t_p2, sa, 0, aux1_before, aux1_after)
        dt_f2 = _get_min_dist(t_f2, sa, 0, aux1_before, aux1_after)
        isi2 = t_f2 - sb[0]
        s2 = dt_p2
        index2 = 0

    # Piecewise-linear integral accumulation.
    # Each segment runs from prev_t to the next event, with value going from
    # y_seg_start to y_seg_end linearly. The contribution is the trapezoid area.
    prev_t = t_start
    y_seg_start = _dist_at_t(isi1, isi2, s1, s2)
    integral = 0.0

    while index1 + index2 < n1 + n2 - 2:
        if (index1 < n1 - 1) and (t_f1 < t_f2 or index2 == n2 - 1):
            # Next event: a spike in train 1 at t_f1.
            index1 += 1
            # End value of the current segment (at t_f1, for train 1's perspective).
            s1_end = dt_f1 * (t_f1 - t_p1) / isi1
            # Interpolated s2 at this event using linear interpolation over isi2.
            s2_at_event = (dt_p2 * (t_f2 - t_f1) + dt_f2 * (t_f1 - t_p2)) / isi2
            y_seg_end = _dist_at_t(isi1, isi2, s1_end, s2_at_event)
            integral += (t_f1 - prev_t) * 0.5 * (y_seg_start + y_seg_end)
            prev_t = t_f1
            # Advance t_p1 to the current event.
            dt_p1 = dt_f1
            t_p1 = t_f1
            # Determine the next following spike in train 1 and update state.
            if index1 < n1 - 1:
                t_f1 = sa[index1 + 1]
                dt_f1 = _get_min_dist(t_f1, sb, index2, aux2_before, aux2_after)
                isi1 = t_f1 - t_p1
                s1 = dt_p1
            else:
                t_f1 = aux1_after
                dt_f1 = dt_p1
                isi1 = _trailing_isi_spike(sa, t_end)
                s1 = dt_p1
            y_seg_start = _dist_at_t(isi1, isi2, s1, s2_at_event)

        elif (index2 < n2 - 1) and (t_f1 > t_f2 or index1 == n1 - 1):
            # Next event: a spike in train 2 at t_f2.
            index2 += 1
            # End value of the current segment.
            s2_end = dt_f2 * (t_f2 - t_p2) / isi2
            # Interpolated s1 at this event.
            s1_at_event = (dt_p1 * (t_f1 - t_f2) + dt_f1 * (t_f2 - t_p1)) / isi1
            y_seg_end = _dist_at_t(isi1, isi2, s1_at_event, s2_end)
            integral += (t_f2 - prev_t) * 0.5 * (y_seg_start + y_seg_end)
            prev_t = t_f2
            # Advance t_p2.
            dt_p2 = dt_f2
            t_p2 = t_f2
            # Determine the next following spike in train 2 and update state.
            if index2 < n2 - 1:
                t_f2 = sb[index2 + 1]
                dt_f2 = _get_min_dist(t_f2, sa, index1, aux1_before, aux1_after)
                isi2 = t_f2 - t_p2
                s2 = dt_p2
            else:
                t_f2 = aux2_after
                dt_f2 = dt_p2
                isi2 = _trailing_isi_spike(sb, t_end)
                s2 = dt_p2
            y_seg_start = _dist_at_t(isi1, isi2, s1_at_event, s2)

        else:
            # Both next spikes coincide exactly (t_f1 == t_f2). At coincident
            # spikes the SPIKE-distance value is 0 by definition (zero distance
            # between the trains at that instant).
            index1 += 1
            index2 += 1
            y_seg_end = 0.0
            integral += (t_f1 - prev_t) * 0.5 * (y_seg_start + y_seg_end)
            prev_t = t_f1
            t_p1 = t_f1
            t_p2 = t_f2
            dt_p1 = 0.0
            dt_p2 = 0.0
            y_seg_start = 0.0
            # Determine next following spikes.
            if index1 < n1 - 1:
                t_f1 = sa[index1 + 1]
                dt_f1 = _get_min_dist(t_f1, sb, index2, aux2_before, aux2_after)
                isi1 = t_f1 - t_p1
            else:
                t_f1 = aux1_after
                dt_f1 = 0.0
                isi1 = _trailing_isi_spike(sa, t_end)
            if index2 < n2 - 1:
                t_f2 = sb[index2 + 1]
                dt_f2 = _get_min_dist(t_f2, sa, index1, aux1_before, aux1_after)
                isi2 = t_f2 - t_p2
            else:
                t_f2 = aux2_after
                dt_f2 = 0.0
                isi2 = _trailing_isi_spike(sb, t_end)

    # Final segment from prev_t to t_end.
    s1_final = dt_f1
    s2_final = dt_f2
    y_seg_end_final = _dist_at_t(isi1, isi2, s1_final, s2_final)
    integral += (t_end - prev_t) * 0.5 * (y_seg_start + y_seg_end_final)

    return integral / (t_end - t_start)
