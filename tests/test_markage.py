import pytest
from hypothesis import given
from hypothesis import strategies as st

from spikedist import van_rossum
from spikedist.van_rossum import _van_rossum_naive

trains = st.lists(
    st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    max_size=12,
)
taus = st.floats(min_value=0.05, max_value=5.0, allow_nan=False, allow_infinity=False)


@given(trains, trains, taus)
def test_markage_matches_naive(a: list[float], b: list[float], tau: float) -> None:
    # The fast and naive paths sum the same terms in different orders, so they agree to
    # floating-point accumulation error rather than exactly. A real bug would show an
    # O(1) difference, far above this tolerance.
    fast = van_rossum(a, b, tau=tau)
    naive = _van_rossum_naive(a, b, tau=tau)
    assert fast == pytest.approx(naive, abs=1e-6, rel=1e-6)
