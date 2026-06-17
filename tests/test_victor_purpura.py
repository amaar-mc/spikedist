import pytest

from spikedist import victor_purpura


def test_empty_trains() -> None:
    assert victor_purpura([], [], cost=1.0) == 0.0


def test_insertions_and_deletions() -> None:
    assert victor_purpura([1.0], [], cost=1.0) == 1.0
    assert victor_purpura([], [1.0, 2.0, 3.0], cost=1.0) == 3.0


def test_identical_trains() -> None:
    assert victor_purpura([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], cost=5.0) == 0.0


def test_single_shift_versus_delete_insert() -> None:
    # One spike each: shift costs cost*dt, delete-plus-insert costs 2.
    assert victor_purpura([0.0], [0.5], cost=1.0) == 0.5
    assert victor_purpura([0.0], [0.5], cost=10.0) == 2.0
    assert victor_purpura([0.0], [0.5], cost=0.0) == 0.0


def test_known_multi_spike_case() -> None:
    # Match the nearest spike (shift 0.5) and delete the extra one (cost 1).
    assert victor_purpura([0.0, 1.0], [0.5], cost=1.0) == 1.5


def test_symmetry() -> None:
    a = [0.0, 0.3, 1.1]
    b = [0.1, 0.9]
    assert victor_purpura(a, b, cost=2.0) == pytest.approx(victor_purpura(b, a, cost=2.0))


def test_unsorted_input_is_handled() -> None:
    assert victor_purpura([2.0, 0.0, 1.0], [0.0, 1.0, 2.0], cost=3.0) == 0.0


def test_negative_cost_rejected() -> None:
    with pytest.raises(ValueError):
        victor_purpura([0.0], [1.0], cost=-1.0)


def test_non_finite_rejected() -> None:
    with pytest.raises(ValueError):
        victor_purpura([float("nan")], [0.0], cost=1.0)
