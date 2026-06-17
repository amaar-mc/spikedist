# spikedist

<p align="center">
  <img src="assets/logo.png" alt="spikedist logo" width="160">
</p>

[![PyPI](https://img.shields.io/pypi/v/spikedist)](https://pypi.org/project/spikedist/)
[![CI](https://github.com/amaar-mc/spikedist/actions/workflows/ci.yml/badge.svg)](https://github.com/amaar-mc/spikedist/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

Spike-train distance and similarity metrics in pure Python with zero dependencies. Implements the Victor-Purpura and van Rossum distances, the ISI-distance, the SPIKE-distance, and the Schreiber and Hunter-Milton similarities on plain sequences of spike times.

## Install

```sh
pip install spikedist
```

For the optional NumPy fast path:

```sh
pip install spikedist[fast]
```

## 30-second example

```python
from spikedist import victor_purpura, van_rossum, isi_distance, spike_distance, schreiber, hunter_milton

a = [0.010, 0.025, 0.090]   # spike times in seconds
b = [0.012, 0.030, 0.095]

victor_purpura(a, b, cost=100.0)              # edit distance, cost is the q parameter
van_rossum(a, b, tau=0.012)                   # kernel distance, tau is the time constant
isi_distance(a, b, interval=[0.0, 0.1])       # ISI-distance in [0, 1]
spike_distance(a, b, interval=[0.0, 0.1])     # SPIKE-distance in [0, 1]
schreiber(a, b, sigma=0.010)                  # Gaussian correlation similarity in [0, 1]
hunter_milton(a, b, tau=0.012)                # nearest-neighbor similarity in (0, 1]
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
  but does not implement Victor-Purpura or van Rossum. `spikedist` now also
  implements both the ISI-distance and the SPIKE-distance matching pyspike's
  conventions exactly, with zero dependencies and no NumPy requirement.

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

The kernel sums are computed in O(n + m) time using the Houghton-Kreuz markage
recursion rather than the naive O(n*m) double loop.

With this normalization the distance between an empty train and a single spike is
`sqrt(0.5)`, and as `tau` grows large the distance approaches
`abs(len(a) - len(b)) / sqrt(2)`.

Both distances are true metrics: non-negative, symmetric, zero only between equal
trains, and they satisfy the triangle inequality. These properties are tested.

### ISI-distance

`isi_distance(a, b, *, interval)` measures the dissimilarity between two spike
trains using their instantaneous inter-spike-interval (ISI) functions. At each
time t the ISI function gives the length of the ISI that contains t. Boundary
intervals use an auxiliary ISI: before the first spike in a train with N > 1
it is `max(first_spike - t_start, second_ISI)`; for a single-spike train it is
`first_spike - t_start`. After the last spike in a train with N > 1 it is
`max(t_end - last_spike, last_ISI)`; for a single-spike train it is
`t_end - last_spike`. An empty train is treated as two auxiliary spikes at
`t_start` and `t_end`, giving a constant ISI of `t_end - t_start`.

The pointwise dissimilarity and time-averaged distance are

```
I(t) = |isi_a(t) - isi_b(t)| / max(isi_a(t), isi_b(t))
D_I  = (1 / (t_end - t_start)) * integral_{t_start}^{t_end} I(t) dt
```

The result lies in `[0, 1]`. The `interval` parameter `[t_start, t_end]` is
required and has no default value. The algorithm runs in O(n + m) time.

The edge convention matches pyspike exactly, validated to floating-point
identity (error == 0.0) on all reference cases from pyspike 0.9.0.

Reference: Kreuz T, Haas JS, Morelli A, Abarbanel HDI, Politi A (2007),
"Measuring spike train synchrony," J Neurosci Methods 165:151-161.

### SPIKE-distance

`spike_distance(a, b, *, interval)` measures the dissimilarity between two spike
trains by tracking, at each time t, how far the nearest preceding and following
spikes in each train are from the spikes in the other train, weighted by the local
mean inter-spike-interval context. The resulting time-resolved profile is
piecewise-linear; the scalar SPIKE-distance is its time-average over the interval.

At each time t lying in the ISI `(t_p, t_f)` of train 1, the local spike-time
distance is interpolated linearly between `dt_p` (distance from the preceding
spike to the nearest spike in train 2) and `dt_f` (distance from the following
spike to the nearest spike in train 2):

```
s_1(t) = (dt_p * (t_f - t) + dt_f * (t - t_p)) / isi_1
```

and symmetrically for train 2. The instantaneous value is

```
S(t)  = 0.5 * (s_1(t) * isi_2 + s_2(t) * isi_1) / mean_isi^2
D_S   = (1 / (t_end - t_start)) * integral_{t_start}^{t_end} S(t) dt
```

The result lies in `[0, 1]`: 0 for identical trains. The `interval` parameter
`[t_start, t_end]` is required and has no default value.

The edge convention matches pyspike exactly, validated to floating-point
identity (max error 5.6e-17, one ULP) on all reference cases from pyspike 0.9.0.

Reference: Kreuz T, Chicharro D, Houghton C, Andrzejak RG, Mormann F (2013),
"Monitoring spike train synchrony," J Neurophysiol 109:1457-1472.

### Multi-unit van Rossum

`van_rossum_multiunit(a, b, *, tau, c)` compares two labeled populations of spike trains,
each given as a mapping from unit label to that unit's train. The parameter `c` in
`[0, 1]` sets how much spikes of different units interact: `c = 0` treats the units as
independent (the Euclidean combination of the per-unit distances), `c = 1` ignores the
labels (the pooled van Rossum distance), and a single unit reduces to `van_rossum`. It
reuses the O(n + m) markage cross-sum.

### Schreiber similarity

`schreiber(a, b, *, sigma)` convolves each train with a Gaussian of width `sigma`
and returns the cosine similarity of the filtered signals, in `[0, 1]`. It is 1
for identical trains.

### Hunter-Milton similarity

`hunter_milton(a, b, *, tau)` scores each spike by `exp(-dt / tau)` to its nearest
neighbor in the other train and averages over both trains, giving a value in
`(0, 1]`. It is 1 for identical trains.

By convention both similarities treat two empty trains as identical (1.0) and a
non-empty train against an empty one as fully dissimilar (0.0).

### Pairwise matrices

`pairwise(trains, metric)` builds the full matrix of any metric over a list of
trains. Parameterize the metric with `functools.partial`:

```python
from functools import partial
from spikedist import pairwise, van_rossum

pairwise(trains, partial(van_rossum, tau=0.01))
```

### NumPy fast path (optional)

When NumPy is installed (`pip install spikedist[fast]`), `van_rossum_matrix`
computes the full N x N pairwise van Rossum distance matrix using NumPy
broadcasting on the cross-kernel sums. It is faster than N^2 calls to
`van_rossum` for moderate to large N and returns numerically identical results.

```python
from spikedist import van_rossum_matrix

trains = [[0.0, 0.1], [0.05, 0.2], [0.3]]
matrix = van_rossum_matrix(trains, tau=0.01)
# matrix[i][j] == van_rossum(trains[i], trains[j], tau=0.01)
```

NumPy remains strictly optional. The package imports and all other functions
work with zero dependencies when NumPy is not installed, and `van_rossum_matrix`
is simply not available.

## Roadmap

- Additional spike-train metrics (SPIKE-synchrony).

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
