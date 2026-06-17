import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from spikedist import van_rossum, van_rossum_multiunit

train = st.lists(
    st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False),
    max_size=8,
)
taus = st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)


def test_single_unit_equals_van_rossum() -> None:
    a = {"x": [0.0, 1.0, 2.5]}
    b = {"x": [0.2, 1.1]}
    for c in (0.0, 0.5, 1.0):
        assert van_rossum_multiunit(a, b, tau=1.0, c=c) == pytest.approx(
            van_rossum([0.0, 1.0, 2.5], [0.2, 1.1], tau=1.0)
        )


def test_independent_units_at_c_zero() -> None:
    a = {"u": [0.0, 1.0], "v": [3.0]}
    b = {"u": [0.5], "v": [3.2, 4.0]}
    expected = math.sqrt(
        van_rossum([0.0, 1.0], [0.5], tau=1.0) ** 2 + van_rossum([3.0], [3.2, 4.0], tau=1.0) ** 2
    )
    assert van_rossum_multiunit(a, b, tau=1.0, c=0.0) == pytest.approx(expected)


def test_pooled_at_c_one() -> None:
    a = {"u": [0.0, 1.0], "v": [3.0]}
    b = {"u": [0.5], "v": [3.2, 4.0]}
    pooled = van_rossum([0.0, 1.0, 3.0], [0.5, 3.2, 4.0], tau=1.0)
    assert van_rossum_multiunit(a, b, tau=1.0, c=1.0) == pytest.approx(pooled)


def test_symmetry_and_self_distance() -> None:
    a = {"u": [0.0, 1.2], "v": [2.0, 2.4]}
    b = {"u": [0.3], "v": [2.1, 3.0, 3.5]}
    assert van_rossum_multiunit(a, b, tau=0.7, c=0.4) == pytest.approx(
        van_rossum_multiunit(b, a, tau=0.7, c=0.4)
    )
    assert van_rossum_multiunit(a, a, tau=0.7, c=0.4) == pytest.approx(0.0, abs=1e-6)


def test_missing_units_treated_as_empty() -> None:
    a = {"u": [0.0, 1.0]}
    b = {"u": [0.0, 1.0], "v": [5.0]}
    # only unit v differs, and it is empty vs one spike
    expected = van_rossum([], [5.0], tau=1.0)
    assert van_rossum_multiunit(a, b, tau=1.0, c=0.0) == pytest.approx(expected)


def test_validation() -> None:
    with pytest.raises(ValueError):
        van_rossum_multiunit({"x": [0.0]}, {"x": [1.0]}, tau=0.0, c=0.0)
    with pytest.raises(ValueError):
        van_rossum_multiunit({"x": [0.0]}, {"x": [1.0]}, tau=1.0, c=1.5)


@given(train, train, train, train, taus, st.floats(min_value=0.0, max_value=1.0))
def test_c_endpoints_match_definitions(
    a1: list[float], a2: list[float], b1: list[float], b2: list[float], tau: float, c: float
) -> None:
    a = {"u": a1, "v": a2}
    b = {"u": b1, "v": b2}
    value = van_rossum_multiunit(a, b, tau=tau, c=c)
    assert value >= 0.0
    # c = 0 is the per-unit Euclidean combine; c = 1 is the pooled distance.
    indep = math.sqrt(van_rossum(a1, b1, tau=tau) ** 2 + van_rossum(a2, b2, tau=tau) ** 2)
    pooled = van_rossum(a1 + a2, b1 + b2, tau=tau)
    assert van_rossum_multiunit(a, b, tau=tau, c=0.0) == pytest.approx(indep, abs=1e-6)
    assert van_rossum_multiunit(a, b, tau=tau, c=1.0) == pytest.approx(pooled, abs=1e-6)
