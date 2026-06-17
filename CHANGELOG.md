# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Schreiber correlation measure and Hunter-Milton similarity.
- Multi-unit van Rossum.
- Fast O(n) van Rossum via the Houghton-Kreuz markage recursion.
- Pairwise distance matrices and an optional NumPy fast path.

## [0.1.0]

### Added
- `victor_purpura(a, b, *, cost)`: Victor-Purpura distance via an O(n*m) dynamic program.
- `van_rossum(a, b, *, tau)`: van Rossum distance via the closed form of the exponential-kernel inner products.
- Input validation with clear errors; spikes treated as an unordered set and sorted internally.
- Test suite with exact closed-form reference values and Hypothesis property tests for the metric axioms.
