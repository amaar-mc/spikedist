"""Tests for isi_distance.

Golden reference values were generated with pyspike 0.9.0:

    import pyspike
    st_a = pyspike.SpikeTrain(<spikes>, [0.0, 1.0])
    st_b = pyspike.SpikeTrain(<spikes>, [0.0, 1.0])
    pyspike.isi_distance(st_a, st_b, interval=[0.0, 1.0])

All values match to floating-point identity (error == 0.0) against pyspike.
These constants are embedded so the test suite runs with zero dependencies.
"""
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from spikedist import isi_distance

INTERVAL = [0.0, 1.0]

# ---------------------------------------------------------------------------
# Golden constants derived from pyspike 0.9.0
# ---------------------------------------------------------------------------

# pyspike.isi_distance(SpikeTrain([0.2,0.4,0.6,0.8],[0,1]),
#                      SpikeTrain([0.2,0.4,0.6,0.8],[0,1]), interval=[0,1])
GOLDEN_IDENTICAL = 0.0

# pyspike.isi_distance(SpikeTrain([0.1,...,0.9],[0,1]),
#                      SpikeTrain([0.5],[0,1]), interval=[0,1])
GOLDEN_HIGH_RATE_VS_SINGLE = 0.7999999999999999

# pyspike.isi_distance(SpikeTrain([0.25,0.75],[0,1]),
#                      SpikeTrain([0.2,0.4,0.6,0.8],[0,1]), interval=[0,1])
GOLDEN_TWO_VS_FOUR = 0.5999999999999999

# pyspike.isi_distance(SpikeTrain([],[0,1]),
#                      SpikeTrain([0.3,0.7],[0,1]), interval=[0,1])
GOLDEN_EMPTY_VS_TWO = 0.6000000000000001

# pyspike.isi_distance(SpikeTrain([0.1,0.5],[0,1]),
#                      SpikeTrain([0.4,0.6],[0,1]), interval=[0,1])
GOLDEN_ISI_CROSS = 0.19

# pyspike.isi_distance(SpikeTrain([0.25],[0,1]),
#                      SpikeTrain([0.75],[0,1]), interval=[0,1])
GOLDEN_SINGLE_EACH = 0.3333333333333333

# pyspike.isi_distance(SpikeTrain([0.5],[0,1]),
#                      SpikeTrain([0.2,0.8],[0,1]), interval=[0,1])
GOLDEN_ONE_VS_TWO = 0.1666666666666668


# ---------------------------------------------------------------------------
# Golden reference tests (tolerance 1e-9 as required)
# ---------------------------------------------------------------------------


def test_golden_identical() -> None:
    d = isi_distance([0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8], interval=INTERVAL)
    assert d == pytest.approx(GOLDEN_IDENTICAL, abs=1e-9, rel=1e-9)


def test_golden_high_rate_vs_single() -> None:
    a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    b = [0.5]
    d = isi_distance(a, b, interval=INTERVAL)
    assert d == pytest.approx(GOLDEN_HIGH_RATE_VS_SINGLE, abs=1e-9, rel=1e-9)


def test_golden_two_vs_four() -> None:
    d = isi_distance([0.25, 0.75], [0.2, 0.4, 0.6, 0.8], interval=INTERVAL)
    assert d == pytest.approx(GOLDEN_TWO_VS_FOUR, abs=1e-9, rel=1e-9)


def test_golden_empty_vs_two() -> None:
    d = isi_distance([], [0.3, 0.7], interval=INTERVAL)
    assert d == pytest.approx(GOLDEN_EMPTY_VS_TWO, abs=1e-9, rel=1e-9)


def test_golden_isi_cross() -> None:
    # Trains where the ISI of a is initially equal to b, then diverges as spikes
    # alternate: a=[0.1,0.5], b=[0.4,0.6].
    d = isi_distance([0.1, 0.5], [0.4, 0.6], interval=INTERVAL)
    assert d == pytest.approx(GOLDEN_ISI_CROSS, abs=1e-9, rel=1e-9)


def test_golden_single_each() -> None:
    d = isi_distance([0.25], [0.75], interval=INTERVAL)
    assert d == pytest.approx(GOLDEN_SINGLE_EACH, abs=1e-9, rel=1e-9)


def test_golden_one_vs_two() -> None:
    d = isi_distance([0.5], [0.2, 0.8], interval=INTERVAL)
    assert d == pytest.approx(GOLDEN_ONE_VS_TWO, abs=1e-9, rel=1e-9)


# ---------------------------------------------------------------------------
# Hand-computed example
# ---------------------------------------------------------------------------


def test_hand_computed_single_spike_each() -> None:
    """Verify 1/3 by hand for a=[0.25], b=[0.75], interval=[0,1].

    Edge convention:
    - a has one spike at 0.25 > t_start=0, so N_a=1 =>
        nu_a (before spike) = 0.25 - 0.0 = 0.25.
    - b has one spike at 0.75 > t_start=0, so N_b=1 =>
        nu_b (before spike) = 0.75 - 0.0 = 0.75.

    Segment [0, 0.25]:
        I = |0.25 - 0.75| / max(0.25, 0.75) = 0.50/0.75 = 2/3.

    At t=0.25 (spike in a), trailing ISI for a (N_a=1):
        nu_a = t_end - 0.25 = 1.0 - 0.25 = 0.75.

    Segment [0.25, 0.75]:
        I = |0.75 - 0.75| / max(0.75, 0.75) = 0.

    At t=0.75 (spike in b), trailing ISI for b (N_b=1):
        nu_b = t_end - 0.75 = 1.0 - 0.75 = 0.25.

    Segment [0.75, 1.0]:
        I = |0.75 - 0.25| / max(0.75, 0.25) = 0.50/0.75 = 2/3.

    Integral: (2/3)*0.25 + 0*0.50 + (2/3)*0.25 = (1/6) + 0 + (1/6) = 1/3.
    Average: (1/3) / 1.0 = 1/3.
    """
    d = isi_distance([0.25], [0.75], interval=[0.0, 1.0])
    assert d == pytest.approx(1.0 / 3.0, abs=1e-14)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


def test_identity() -> None:
    a = [0.1, 0.3, 0.5, 0.7, 0.9]
    d = isi_distance(a, a, interval=INTERVAL)
    assert d == 0.0


def test_symmetry() -> None:
    a = [0.25, 0.75]
    b = [0.2, 0.4, 0.6, 0.8]
    d_ab = isi_distance(a, b, interval=INTERVAL)
    d_ba = isi_distance(b, a, interval=INTERVAL)
    assert d_ab == pytest.approx(d_ba, abs=1e-14)


def test_range() -> None:
    d = isi_distance([0.1, 0.2, 0.3], [0.7, 0.9], interval=INTERVAL)
    assert 0.0 <= d <= 1.0


def test_empty_both() -> None:
    # Both empty -> same ISI everywhere -> distance 0.
    d = isi_distance([], [], interval=INTERVAL)
    assert d == 0.0


def test_empty_symmetry() -> None:
    d1 = isi_distance([], [0.3, 0.7], interval=INTERVAL)
    d2 = isi_distance([0.3, 0.7], [], interval=INTERVAL)
    assert d1 == pytest.approx(d2, abs=1e-14)


def test_unsorted_input_accepted() -> None:
    # Spikes are sorted internally.
    d_sorted = isi_distance([0.1, 0.5], [0.4, 0.6], interval=INTERVAL)
    d_unsorted = isi_distance([0.5, 0.1], [0.6, 0.4], interval=INTERVAL)
    assert d_sorted == pytest.approx(d_unsorted, abs=1e-14)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_interval_start_ge_end_raises() -> None:
    with pytest.raises(ValueError):
        isi_distance([0.5], [0.5], interval=[1.0, 0.0])


def test_interval_equal_raises() -> None:
    with pytest.raises(ValueError):
        isi_distance([], [], interval=[0.5, 0.5])


def test_non_finite_spike_raises() -> None:
    with pytest.raises(ValueError):
        isi_distance([float("inf")], [0.5], interval=INTERVAL)


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------

trains_hyp = st.lists(
    st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    max_size=8,
)


@given(trains_hyp)
def test_self_distance_is_zero(spikes: list[float]) -> None:
    assert isi_distance(spikes, spikes, interval=INTERVAL) == 0.0


@given(trains_hyp, trains_hyp)
def test_non_negative_and_symmetric(a: list[float], b: list[float]) -> None:
    d_ab = isi_distance(a, b, interval=INTERVAL)
    d_ba = isi_distance(b, a, interval=INTERVAL)
    assert d_ab >= 0.0
    assert math.isclose(d_ab, d_ba, rel_tol=1e-12, abs_tol=1e-12)


@given(trains_hyp, trains_hyp)
def test_bounded_in_unit_interval(a: list[float], b: list[float]) -> None:
    d = isi_distance(a, b, interval=INTERVAL)
    assert 0.0 <= d <= 1.0 + 1e-12
