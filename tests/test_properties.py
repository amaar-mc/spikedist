import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from spikedist import hunter_milton, schreiber, van_rossum, victor_purpura

trains = st.lists(
    st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    max_size=8,
)

COST = 1.5
TAU = 0.7
SIGMA = 0.7


@given(trains)
def test_self_distance_is_zero(a: list[float]) -> None:
    assert victor_purpura(a, a, cost=COST) == 0.0
    # van_rossum's self-distance is mathematically zero, but the self and cross kernel
    # sums are accumulated in different orders, so it reaches zero only to within
    # floating-point error.
    assert van_rossum(a, a, tau=TAU) == pytest.approx(0.0, abs=1e-6)


@given(trains)
def test_self_similarity_is_one(a: list[float]) -> None:
    assert schreiber(a, a, sigma=SIGMA) == pytest.approx(1.0)
    assert hunter_milton(a, a, tau=TAU) == pytest.approx(1.0)


@given(trains, trains)
def test_distances_non_negative_and_symmetric(a: list[float], b: list[float]) -> None:
    for d_ab, d_ba in (
        (victor_purpura(a, b, cost=COST), victor_purpura(b, a, cost=COST)),
        (van_rossum(a, b, tau=TAU), van_rossum(b, a, tau=TAU)),
    ):
        assert d_ab >= 0.0
        assert math.isclose(d_ab, d_ba, rel_tol=1e-9, abs_tol=1e-9)


@given(trains, trains)
def test_similarities_bounded_and_symmetric(a: list[float], b: list[float]) -> None:
    for s_ab, s_ba in (
        (schreiber(a, b, sigma=SIGMA), schreiber(b, a, sigma=SIGMA)),
        (hunter_milton(a, b, tau=TAU), hunter_milton(b, a, tau=TAU)),
    ):
        assert 0.0 <= s_ab <= 1.0
        assert math.isclose(s_ab, s_ba, rel_tol=1e-9, abs_tol=1e-9)


@given(trains, trains, trains)
def test_triangle_inequality(a: list[float], b: list[float], c: list[float]) -> None:
    ab = victor_purpura(a, b, cost=COST)
    bc = victor_purpura(b, c, cost=COST)
    ac = victor_purpura(a, c, cost=COST)
    assert ac <= ab + bc + 1e-9

    vab = van_rossum(a, b, tau=TAU)
    vbc = van_rossum(b, c, tau=TAU)
    vac = van_rossum(a, c, tau=TAU)
    assert vac <= vab + vbc + 1e-9
