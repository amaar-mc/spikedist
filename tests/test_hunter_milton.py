import math

import pytest

from spikedist import hunter_milton


def test_identical_is_one() -> None:
    assert hunter_milton([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], tau=0.5) == pytest.approx(1.0)


def test_single_spikes_closed_form() -> None:
    d, tau = 1.0, 2.0
    assert hunter_milton([0.0], [d], tau=tau) == pytest.approx(math.exp(-d / tau))


def test_nearest_neighbor_is_used() -> None:
    # a's spike at 0 maps to the nearer of b's two spikes (0.1, distance 0.1).
    tau = 1.0
    a_to_b = math.exp(-0.1 / tau)
    b_to_a = (math.exp(-0.3 / tau) + math.exp(-0.1 / tau)) / 2
    expected = 0.5 * (a_to_b + b_to_a)
    assert hunter_milton([0.0], [-0.3, 0.1], tau=tau) == pytest.approx(expected)


def test_symmetry() -> None:
    a = [0.0, 0.4, 1.2]
    b = [0.2, 1.0]
    assert hunter_milton(a, b, tau=0.5) == pytest.approx(hunter_milton(b, a, tau=0.5))


def test_range() -> None:
    s = hunter_milton([0.0, 0.5], [0.3, 2.0], tau=0.4)
    assert 0.0 < s <= 1.0


def test_empty_conventions() -> None:
    assert hunter_milton([], [], tau=1.0) == 1.0
    assert hunter_milton([0.0], [], tau=1.0) == 0.0
    assert hunter_milton([], [1.0], tau=1.0) == 0.0


def test_tau_must_be_positive() -> None:
    with pytest.raises(ValueError):
        hunter_milton([0.0], [1.0], tau=0.0)
