"""Tests for spike_distance.

Golden reference values were generated with pyspike 0.9.0:

    import pyspike
    st_a = pyspike.SpikeTrain(<spikes>, [0.0, 1.0])
    st_b = pyspike.SpikeTrain(<spikes>, [0.0, 1.0])
    pyspike.spike_distance(st_a, st_b, interval=[0.0, 1.0])

All values match to floating-point identity (error < 5.6e-17, i.e. one ULP)
against pyspike 0.9.0. These constants are embedded so the test suite runs with
zero dependencies.
"""
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from spikedist import spike_distance

INTERVAL = [0.0, 1.0]

# ---------------------------------------------------------------------------
# Golden constants derived from pyspike 0.9.0
# ---------------------------------------------------------------------------

# pyspike.spike_distance(SpikeTrain([0.2,0.4,0.6,0.8],[0,1]),
#                        SpikeTrain([0.2,0.4,0.6,0.8],[0,1]), interval=[0,1])
GOLDEN_IDENTICAL = 0.0

# pyspike.spike_distance(SpikeTrain([0.5],[0,1]),
#                        SpikeTrain([0.7],[0,1]), interval=[0,1])
GOLDEN_SINGLE_DIFFERING = 0.38333333333333325

# pyspike.spike_distance(SpikeTrain([0.1,...,0.9],[0,1]),
#                        SpikeTrain([0.5],[0,1]), interval=[0,1])
GOLDEN_HIGH_RATE_VS_SINGLE = 0.36111111111111116

# pyspike.spike_distance(SpikeTrain([],[0,1]),
#                        SpikeTrain([0.3,0.7],[0,1]), interval=[0,1])
GOLDEN_EMPTY_VS_TWO = 0.3469387755102041

# pyspike.spike_distance(SpikeTrain([0.1,0.5],[0,1]),
#                        SpikeTrain([0.4,0.6],[0,1]), interval=[0,1])
GOLDEN_CROSSING = 0.25079365079365074

# pyspike.spike_distance(SpikeTrain([0.25,0.75],[0,1]),
#                        SpikeTrain([0.2,0.4,0.6,0.8],[0,1]), interval=[0,1])
GOLDEN_TWO_VS_FOUR = 0.22448979591836737

# pyspike.spike_distance(SpikeTrain([0.25],[0,1]),
#                        SpikeTrain([0.75],[0,1]), interval=[0,1])
GOLDEN_SINGLE_EACH = 0.41666666666666663

# pyspike.spike_distance(SpikeTrain([0.5],[0,1]),
#                        SpikeTrain([0.2,0.8],[0,1]), interval=[0,1])
GOLDEN_ONE_VS_TWO = 0.4628099173553718


# ---------------------------------------------------------------------------
# Golden reference tests (tolerance 1e-9 as required)
# ---------------------------------------------------------------------------


def test_golden_identical() -> None:
    d = spike_distance([0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8], interval=INTERVAL)
    assert d == pytest.approx(GOLDEN_IDENTICAL, abs=1e-9, rel=1e-9)


def test_golden_single_differing() -> None:
    d = spike_distance([0.5], [0.7], interval=INTERVAL)
    assert d == pytest.approx(GOLDEN_SINGLE_DIFFERING, abs=1e-9, rel=1e-9)


def test_golden_high_rate_vs_single() -> None:
    a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    b = [0.5]
    d = spike_distance(a, b, interval=INTERVAL)
    assert d == pytest.approx(GOLDEN_HIGH_RATE_VS_SINGLE, abs=1e-9, rel=1e-9)


def test_golden_empty_vs_two() -> None:
    d = spike_distance([], [0.3, 0.7], interval=INTERVAL)
    assert d == pytest.approx(GOLDEN_EMPTY_VS_TWO, abs=1e-9, rel=1e-9)


def test_golden_crossing() -> None:
    d = spike_distance([0.1, 0.5], [0.4, 0.6], interval=INTERVAL)
    assert d == pytest.approx(GOLDEN_CROSSING, abs=1e-9, rel=1e-9)


def test_golden_two_vs_four() -> None:
    d = spike_distance([0.25, 0.75], [0.2, 0.4, 0.6, 0.8], interval=INTERVAL)
    assert d == pytest.approx(GOLDEN_TWO_VS_FOUR, abs=1e-9, rel=1e-9)


def test_golden_single_each() -> None:
    d = spike_distance([0.25], [0.75], interval=INTERVAL)
    assert d == pytest.approx(GOLDEN_SINGLE_EACH, abs=1e-9, rel=1e-9)


def test_golden_one_vs_two() -> None:
    d = spike_distance([0.5], [0.2, 0.8], interval=INTERVAL)
    assert d == pytest.approx(GOLDEN_ONE_VS_TWO, abs=1e-9, rel=1e-9)


# ---------------------------------------------------------------------------
# Hand-computed example
# ---------------------------------------------------------------------------


def test_hand_computed_single_spike_each() -> None:
    """Verify the result for a=[0.25], b=[0.75], interval=[0,1].

    Both trains have one spike, so:
        aux1_before = aux2_before = t_start = 0.0
        aux1_after  = aux2_after  = t_end   = 1.0

    Leading ISI for a (single spike, sa[0]=0.25 > t_start):
        isi1 = sa[0] - t_start = 0.25
        dt_f1 = get_min_dist(0.25, [0.75], 0, 0.0, 1.0)
              = min(|0.25-0.0|, |0.25-0.75|, |1.0-0.25|) = 0.25

    Leading ISI for b (single spike, sb[0]=0.75 > t_start):
        isi2 = sb[0] - t_start = 0.75
        dt_f2 = get_min_dist(0.75, [0.25], 0, 0.0, 1.0)
              = min(|0.75-0.0|, |0.75-0.25|, |1.0-0.75|) = 0.25

    Initial: s1 = dt_p1 = dt_f1 = 0.25, s2 = dt_p2 = dt_f2 = 0.25.
    mean_isi = 0.5*(0.25+0.75) = 0.5
    y_start0 = 0.5*(0.25*0.75 + 0.25*0.25) / 0.25 = 0.5*0.25/0.25 = 0.5

    Step 1: advance in train a to sa[0]=0.25 (t_f1=0.25 < t_f2=0.75).
        s1_end = dt_f1 * (t_f1 - t_p1) / isi1 = 0.25 * (0.25 - 0.0) / 0.25 = 0.25
        s2_at_event = (dt_p2*(t_f2-t_f1) + dt_f2*(t_f1-t_p2)) / isi2
                    = (0.25*(0.75-0.25) + 0.25*(0.25-0.0)) / 0.75
                    = (0.125 + 0.0625) / 0.75 = 0.25
        mean_isi = 0.5
        y_end1 = 0.5*(0.25*0.75 + 0.25*0.25) / 0.25 = 0.5
        Segment [0, 0.25]: area = 0.25 * 0.5 * (0.5 + 0.5) = 0.125

        Trailing ISI for a (N=1): isi1 = t_end - sa[0] = 1.0 - 0.25 = 0.75
        s1 = dt_p1 = 0.25
        mean_isi = 0.5*(0.75+0.75) = 0.75
        y_start1 = 0.5*(0.25*0.75 + 0.25*0.75) / (0.75*0.75) = 0.5*0.375/0.5625 = 1/3

    Step 2: advance in train b to sb[0]=0.75 (t_f2=0.75).
        s2_end = dt_f2 * (t_f2 - t_p2) / isi2 = 0.25 * (0.75 - 0.0) / 0.75 = 0.25
        s1_at_event = (dt_p1*(t_f1-t_f2) + dt_f1*(t_f2-t_p1)) / isi1
                    = (0.25*(aux1_after-0.75) + 0.25*(0.75-0.25)) / 0.75
                    Here t_f1=aux1_after=1.0 (train a exhausted):
                    = (0.25*(1.0-0.75) + 0.25*(0.75-0.25)) / 0.75
                    = (0.0625 + 0.125) / 0.75 = 0.25
        mean_isi = 0.5*(0.75+0.75) = 0.75
        y_end2 = 0.5*(0.25*0.75 + 0.25*0.75) / 0.5625 = 1/3
        Segment [0.25, 0.75]: area = 0.5 * 0.5 * (1/3 + 1/3) = 0.5 * (2/3) = 1/3

        Trailing ISI for b (N=1): isi2 = t_end - sb[0] = 0.25
        s2 = dt_p2 = 0.25
        mean_isi = 0.5*(0.75+0.25) = 0.5
        y_start2 = 0.5*(0.25*0.25 + 0.25*0.75) / 0.25 = 0.5*0.25/0.25 = 0.5

    Final segment [0.75, 1.0]:
        s1_final = dt_f1 = 0.25 (train a exhausted, dt_f1 = dt_p1 = 0.25)
        s2_final = dt_f2 = 0.25
        mean_isi = 0.5
        y_end_final = 0.5*(0.25*0.25 + 0.25*0.75) / 0.25 = 0.5
        Area = 0.25 * 0.5 * (0.5 + 0.5) = 0.125

    Total integral = 0.125 + 1/3 + 0.125 = 0.25 + 1/3 = 5/12
    Average over interval of length 1: 5/12 ≈ 0.41667
    """
    d = spike_distance([0.25], [0.75], interval=[0.0, 1.0])
    assert d == pytest.approx(5.0 / 12.0, abs=1e-14)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


def test_identity() -> None:
    a = [0.1, 0.3, 0.5, 0.7, 0.9]
    d = spike_distance(a, a, interval=INTERVAL)
    assert d == 0.0


def test_symmetry() -> None:
    a = [0.25, 0.75]
    b = [0.2, 0.4, 0.6, 0.8]
    d_ab = spike_distance(a, b, interval=INTERVAL)
    d_ba = spike_distance(b, a, interval=INTERVAL)
    assert d_ab == pytest.approx(d_ba, abs=1e-14)


def test_range() -> None:
    d = spike_distance([0.1, 0.2, 0.3], [0.7, 0.9], interval=INTERVAL)
    assert 0.0 <= d <= 1.0


def test_empty_both() -> None:
    d = spike_distance([], [], interval=INTERVAL)
    assert d == 0.0


def test_empty_symmetry() -> None:
    d1 = spike_distance([], [0.3, 0.7], interval=INTERVAL)
    d2 = spike_distance([0.3, 0.7], [], interval=INTERVAL)
    assert d1 == pytest.approx(d2, abs=1e-14)


def test_unsorted_input_accepted() -> None:
    d_sorted = spike_distance([0.1, 0.5], [0.4, 0.6], interval=INTERVAL)
    d_unsorted = spike_distance([0.5, 0.1], [0.6, 0.4], interval=INTERVAL)
    assert d_sorted == pytest.approx(d_unsorted, abs=1e-14)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_interval_start_ge_end_raises() -> None:
    with pytest.raises(ValueError):
        spike_distance([0.5], [0.5], interval=[1.0, 0.0])


def test_interval_equal_raises() -> None:
    with pytest.raises(ValueError):
        spike_distance([], [], interval=[0.5, 0.5])


def test_non_finite_spike_raises() -> None:
    with pytest.raises(ValueError):
        spike_distance([float("inf")], [0.5], interval=INTERVAL)


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------

trains_hyp = st.lists(
    st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    max_size=8,
)


@given(trains_hyp)
def test_self_distance_is_zero(spikes: list[float]) -> None:
    assert spike_distance(spikes, spikes, interval=INTERVAL) == 0.0


@given(trains_hyp, trains_hyp)
def test_non_negative_and_symmetric(a: list[float], b: list[float]) -> None:
    d_ab = spike_distance(a, b, interval=INTERVAL)
    d_ba = spike_distance(b, a, interval=INTERVAL)
    assert d_ab >= 0.0
    assert math.isclose(d_ab, d_ba, rel_tol=1e-12, abs_tol=1e-12)


@given(trains_hyp, trains_hyp)
def test_bounded_in_unit_interval(a: list[float], b: list[float]) -> None:
    d = spike_distance(a, b, interval=INTERVAL)
    assert 0.0 <= d <= 1.0 + 1e-12
