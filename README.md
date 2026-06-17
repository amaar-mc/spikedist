# spikedist

[![CI](https://github.com/amaar-mc/spikedist/actions/workflows/ci.yml/badge.svg)](https://github.com/amaar-mc/spikedist/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

Spike-train distance metrics in pure Python with zero dependencies. Implements the Victor-Purpura and van Rossum distances on plain sequences of spike times.

## Install

```sh
pip install spikedist
```

## 30-second example

```python
from spikedist import victor_purpura, van_rossum

a = [0.010, 0.025, 0.090]   # spike times in seconds
b = [0.012, 0.030, 0.095]

victor_purpura(a, b, cost=100.0)  # edit distance, cost is the q parameter
van_rossum(a, b, tau=0.012)       # kernel distance, tau is the time constant
```

Spike times can be Python lists, tuples, or any sequence of numbers, including
NumPy arrays. They are treated as an unordered set of event times and sorted
internally. There is no NumPy requirement.

## Why this exists

The Victor-Purpura and van Rossum distances are two of the most cited spike-train
metrics, but every Python implementation lives inside a heavy framework or a
compiled extension:

- `elephant` implements both, but requires `neo` and `quantities` and works on
  `neo.SpikeTrain` objects with units.
- `pymuvr` is a fast multi-unit van Rossum implementation, but is a C++ extension
  and requires NumPy.
- `pyspike` is excellent for ISI-distance, SPIKE-distance, and SPIKE-synchrony,
  but does not implement Victor-Purpura or van Rossum.

`spikedist` is a small, typed, dependency-free package for when you just want the
distance between two spike trains.

## Definitions

### Victor-Purpura

`victor_purpura(a, b, *, cost)` is the minimum total cost to turn train `a` into
train `b` using three operations: insert a spike (cost 1), delete a spike
(cost 1), and shift a spike by `dt` (cost `cost * abs(dt)`). `cost` is the
parameter usually written `q`. It is computed with an O(n*m) dynamic program.

- `cost = 0` counts only the difference in spike count.
- As `cost` grows, shifting becomes expensive and each unmatched spike approaches
  a cost of 2.

### van Rossum

`van_rossum(a, b, *, tau)` convolves each train with a causal exponential kernel
`exp(-t / tau)` and returns the Euclidean distance between the filtered signals.
Using the closed form of the kernel inner products,

```
D^2 = 0.5 * (Saa + Sbb - 2 * Sab),  Sxy = sum over spike pairs exp(-|xi - yj| / tau)
```

With this normalization the distance between an empty train and a single spike is
`sqrt(0.5)`, and as `tau` grows large the distance approaches
`abs(len(a) - len(b)) / sqrt(2)`.

Both functions are true metrics: non-negative, symmetric, zero only between equal
trains, and they satisfy the triangle inequality. These properties are tested.

## Roadmap

- Schreiber correlation measure and Hunter-Milton similarity.
- Multi-unit van Rossum.
- Fast O(n) van Rossum via the Houghton-Kreuz markage recursion.
- Pairwise distance matrices and an optional NumPy fast path.

## Testing

```sh
pip install -e ".[dev]"
pytest
```

Tests cover exact closed-form reference values and metric-property invariants
(identity, symmetry, non-negativity, triangle inequality) via Hypothesis.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

MIT. See [LICENSE](./LICENSE).
