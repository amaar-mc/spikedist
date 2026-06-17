# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Multi-unit van Rossum.
- Fast O(n) van Rossum via the Houghton-Kreuz markage recursion.
- An optional NumPy fast path for large pairwise computations.

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
