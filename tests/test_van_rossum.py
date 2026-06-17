import math

import pytest

from spikedist import van_rossum


def test_empty_trains() -> None:
    assert van_rossum([], [], tau=1.0) == 0.0


def test_empty_versus_single_spike() -> None:
    assert van_rossum([0.0], [], tau=1.0) == pytest.approx(math.sqrt(0.5))
    assert van_rossum([], [3.0], tau=2.0) == pytest.approx(math.sqrt(0.5))


def test_identical_trains() -> None:
    assert van_rossum([0.0, 1.0], [0.0, 1.0], tau=1.0) == 0.0


def test_two_single_spikes_closed_form() -> None:
    # For one spike each, D = sqrt(1 - exp(-d / tau)).
    d, tau = 1.0, 1.0
    assert van_rossum([0.0], [d], tau=tau) == pytest.approx(math.sqrt(1.0 - math.exp(-d / tau)))


def test_large_tau_approaches_count_difference() -> None:
    # As tau grows large, D approaches abs(len(a) - len(b)) / sqrt(2).
    assert van_rossum([0.0, 1.0], [0.5], tau=1e9) == pytest.approx(1.0 / math.sqrt(2.0), abs=1e-4)


def test_symmetry() -> None:
    a = [0.0, 0.4, 1.2]
    b = [0.2, 1.0]
    assert van_rossum(a, b, tau=0.5) == pytest.approx(van_rossum(b, a, tau=0.5))


def test_unsorted_input_is_handled() -> None:
    assert van_rossum([1.0, 0.0], [0.0, 1.0], tau=1.0) == 0.0


def test_tau_must_be_positive() -> None:
    with pytest.raises(ValueError):
        van_rossum([0.0], [1.0], tau=0.0)


def test_non_finite_rejected() -> None:
    with pytest.raises(ValueError):
        van_rossum([0.0], [float("inf")], tau=1.0)
