# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0]

### Added
- `isi_distance(a, b, *, interval)`: ISI-distance (Kreuz et al. 2007) between two spike trains over a finite time interval. For each train the instantaneous inter-spike-interval function assigns to time t the length of the ISI that contains t; boundary ISIs are handled with the pyspike edge convention (auxiliary ISI before the first spike is `max(first_spike - t_start, second_ISI)` for N > 1, or `first_spike - t_start` for N == 1; auxiliary ISI after the last spike is `max(t_end - last_spike, last_ISI)` for N > 1, or `t_end - last_spike` for N == 1; empty trains are treated as [t_start, t_end]). The distance is `integral |isi_a(t) - isi_b(t)| / max(isi_a(t), isi_b(t)) dt` normalized by the interval length, giving a value in [0, 1]. The `interval` parameter is required (no default). Validated against pyspike 0.9.0 to floating-point identity (error == 0.0) on all reference cases.

## [0.5.0]

### Added
- `van_rossum_matrix(trains, *, tau)`: NumPy-backed pairwise van Rossum distance matrix. Computes the full N x N matrix for a list of spike trains using NumPy broadcasting on the cross-kernel sums, which is faster than N^2 calls to `van_rossum` for moderate to large N. Results are numerically identical to `pairwise(trains, partial(van_rossum, tau=tau))` on all off-diagonal entries (tolerance 1e-9); diagonal entries are exactly 0.0.
- `fast` optional-dependency extra: `pip install spikedist[fast]` installs `numpy>=1.21`. NumPy remains strictly optional; the package imports and runs with zero dependencies when NumPy is not installed, and `van_rossum_matrix` is simply not available.

## [0.4.0]

### Added
- `van_rossum_multiunit(a, b, *, tau, c)`: multi-unit (population) van Rossum distance over labeled spike trains, with a cross-unit interaction parameter `c`. At `c = 0` it is the Euclidean combination of per-unit distances; at `c = 1` it equals the pooled van Rossum distance; with a single unit it equals `van_rossum`. Pure Python, reusing the O(n + m) markage cross-sum.

## [0.3.0]

### Changed
- `van_rossum` now computes its kernel sums in O(n + m) time using the Houghton-Kreuz markage recursion instead of the naive O(n*m) double loop. Results are unchanged; a property test checks the fast path against the naive reference across random trains.

## [0.2.0]

### Added
- `schreiber(a, b, *, sigma)`: Gaussian correlation similarity in [0, 1].
- `hunter_milton(a, b, *, tau)`: nearest-neighbor similarity in (0, 1].
- `pairwise(trains, metric)`: full matrix of any metric over a list of trains.

## [0.1.0]

### Added
- `victor_purpura(a, b, *, cost)`: Victor-Purpura distance via an O(n*m) dynamic program.
- `van_rossum(a, b, *, tau)`: van Rossum distance via the closed form of the exponential-kernel inner products.
- Input validation with clear errors; spikes treated as an unordered set and sorted internally.
- Test suite with exact closed-form reference values and Hypothesis property tests for the metric axioms.
