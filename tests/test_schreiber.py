import math

import pytest

from spikedist import schreiber


def test_identical_is_one() -> None:
    assert schreiber([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], sigma=0.5) == pytest.approx(1.0)


def test_single_spikes_closed_form() -> None:
    d, sigma = 1.0, 0.5
    expected = math.exp(-(d * d) / (4 * sigma * sigma))
    assert schreiber([0.0], [d], sigma=sigma) == pytest.approx(expected)


def test_symmetry() -> None:
    a = [0.0, 0.4, 1.2]
    b = [0.2, 1.0]
    assert schreiber(a, b, sigma=0.3) == pytest.approx(schreiber(b, a, sigma=0.3))


def test_range() -> None:
    s = schreiber([0.0, 0.5], [0.3, 2.0, 2.5], sigma=0.4)
    assert 0.0 <= s <= 1.0


def test_empty_conventions() -> None:
    assert schreiber([], [], sigma=1.0) == 1.0
    assert schreiber([0.0], [], sigma=1.0) == 0.0
    assert schreiber([], [1.0], sigma=1.0) == 0.0


def test_unsorted_input_is_handled() -> None:
    assert schreiber([1.0, 0.0], [0.0, 1.0], sigma=0.5) == pytest.approx(1.0)


def test_sigma_must_be_positive() -> None:
    with pytest.raises(ValueError):
        schreiber([0.0], [1.0], sigma=0.0)
