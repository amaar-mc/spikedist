from functools import partial

import pytest

from spikedist import pairwise, van_rossum, victor_purpura


def test_matrix_shape_and_diagonal() -> None:
    trains = [[0.0, 1.0], [0.5], [2.0, 3.0, 4.0]]
    m = pairwise(trains, partial(victor_purpura, cost=1.0))
    assert len(m) == 3
    assert all(len(row) == 3 for row in m)
    for i in range(3):
        assert m[i][i] == 0.0  # distance to self


def test_matches_direct_calls() -> None:
    trains = [[0.0], [1.0]]
    m = pairwise(trains, partial(van_rossum, tau=0.5))
    assert m[0][1] == pytest.approx(van_rossum([0.0], [1.0], tau=0.5))
    assert m[1][0] == pytest.approx(van_rossum([1.0], [0.0], tau=0.5))


def test_empty_list() -> None:
    assert pairwise([], partial(victor_purpura, cost=1.0)) == []
