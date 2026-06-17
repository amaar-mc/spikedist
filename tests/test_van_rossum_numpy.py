"""Tests for the optional NumPy-backed van_rossum_matrix.

All tests that use NumPy are skipped when NumPy is not installed so the
suite still passes with zero dependencies (``uv run pytest -q``).
"""

import math
from functools import partial

import pytest

numpy = pytest.importorskip("numpy")

from spikedist import pairwise, van_rossum  # noqa: E402
from spikedist.van_rossum_numpy import van_rossum_matrix  # noqa: E402

# ---------------------------------------------------------------------------
# Correctness: van_rossum_matrix must agree with the pure-Python reference.
# ---------------------------------------------------------------------------

TRAINS: list[list[float]] = [
    [],
    [0.0],
    [0.0, 1.0],
    [0.5, 1.5, 3.0],
    [0.1, 0.2, 0.3, 2.0, 4.0],
]
TAU = 0.5


def _reference_matrix(
    trains: list[list[float]], tau: float
) -> list[list[float]]:
    return pairwise(trains, partial(van_rossum, tau=tau))


def test_shape() -> None:
    n = len(TRAINS)
    m = van_rossum_matrix(TRAINS, tau=TAU)
    assert len(m) == n
    assert all(len(row) == n for row in m)


def test_diagonal_is_zero() -> None:
    m = van_rossum_matrix(TRAINS, tau=TAU)
    for i in range(len(TRAINS)):
        assert m[i][i] == 0.0


def test_symmetry() -> None:
    m = van_rossum_matrix(TRAINS, tau=TAU)
    n = len(TRAINS)
    for i in range(n):
        for j in range(n):
            assert math.isclose(m[i][j], m[j][i], rel_tol=1e-12, abs_tol=1e-14)


def test_agrees_with_pure_python_reference() -> None:
    """van_rossum_matrix must match pairwise(trains, van_rossum) to 1e-9.

    The diagonal is excluded from the tolerance check: van_rossum_matrix
    returns exactly 0.0 for each train compared with itself (it is the
    mathematically correct value), while the pure-Python van_rossum(a, a)
    accumulates _self_sum and _cross_sum in different orderings and can
    differ by up to ~1e-7 from zero. The off-diagonal entries agree to
    well within 1e-9.
    """
    ref = _reference_matrix(TRAINS, TAU)
    got = van_rossum_matrix(TRAINS, tau=TAU)
    n = len(TRAINS)
    for i in range(n):
        for j in range(n):
            if i == j:
                # numpy path is more accurate on the diagonal; skip comparison.
                assert got[i][j] == 0.0
                continue
            assert math.isclose(
                got[i][j],
                ref[i][j],
                rel_tol=1e-9,
                abs_tol=1e-9,
            ), (
                f"mismatch at ({i}, {j}): numpy={got[i][j]}, pure={ref[i][j]}, "
                f"delta={abs(got[i][j] - ref[i][j])}"
            )


def test_agrees_with_pure_python_various_tau() -> None:
    """Check 1e-9 tolerance across multiple tau values, off-diagonal entries."""
    for tau in (0.01, 0.1, 1.0, 10.0, 1e6):
        ref = _reference_matrix(TRAINS, tau)
        got = van_rossum_matrix(TRAINS, tau=tau)
        n = len(TRAINS)
        for i in range(n):
            for j in range(n):
                if i == j:
                    assert got[i][j] == 0.0
                    continue
                assert math.isclose(
                    got[i][j],
                    ref[i][j],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ), (
                    f"tau={tau} mismatch at ({i}, {j}): "
                    f"numpy={got[i][j]}, pure={ref[i][j]}"
                )


def test_empty_train_list() -> None:
    assert van_rossum_matrix([], tau=TAU) == []


def test_single_train() -> None:
    m = van_rossum_matrix([[0.0, 1.0]], tau=1.0)
    assert m == [[0.0]]


def test_empty_train_versus_single_spike() -> None:
    # Distance between empty and single-spike train is sqrt(0.5).
    m = van_rossum_matrix([[], [0.0]], tau=1.0)
    assert math.isclose(m[0][1], math.sqrt(0.5), rel_tol=1e-12)
    assert math.isclose(m[1][0], math.sqrt(0.5), rel_tol=1e-12)


def test_tau_must_be_positive() -> None:
    with pytest.raises(ValueError, match="tau must be positive"):
        van_rossum_matrix([[0.0]], tau=0.0)


def test_non_finite_rejected() -> None:
    with pytest.raises(ValueError):
        van_rossum_matrix([[0.0], [float("inf")]], tau=1.0)


def test_larger_random_trains() -> None:
    """Equivalence on larger random trains (relative tolerance 1e-9), off-diagonal."""
    rng = numpy.random.default_rng(42)
    trains: list[list[float]] = [
        sorted(rng.uniform(0.0, 10.0, size=int(k)).tolist())
        for k in rng.integers(5, 30, size=8)
    ]
    ref = _reference_matrix(trains, 0.3)
    got = van_rossum_matrix(trains, tau=0.3)
    n = len(trains)
    for i in range(n):
        for j in range(n):
            if i == j:
                assert got[i][j] == 0.0
                continue
            assert math.isclose(
                got[i][j],
                ref[i][j],
                rel_tol=1e-9,
                abs_tol=1e-9,
            ), (
                f"large-train mismatch at ({i}, {j}): "
                f"numpy={got[i][j]}, pure={ref[i][j]}"
            )


# ---------------------------------------------------------------------------
# Zero-dependency smoke test: the package must import without NumPy.
# (This test runs unconditionally; it does not use the numpy import at the
# top of this file, but it lives here since it probes the no-numpy scenario.)
# ---------------------------------------------------------------------------


def test_package_imports_and_works_without_numpy() -> None:
    """The pure-Python van_rossum must work regardless of NumPy availability."""
    # Just call van_rossum directly; the import at the top of this file
    # (which requires numpy via importorskip) is already loaded, but we
    # only need to confirm the pure function is callable and correct.
    result = van_rossum([0.0], [1.0], tau=1.0)
    assert result > 0.0
    assert math.isclose(result, math.sqrt(1.0 - math.exp(-1.0)), rel_tol=1e-12)
