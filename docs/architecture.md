# Architecture

`spikedist` is a small set of pure functions, one module per metric, sharing a
single input-validation helper. There is no shared state and no I/O.

## Input handling

`_validate.check_train` converts any sequence of numbers to a list of floats,
rejects non-finite values with a clear error, and sorts ascending. Spikes are
treated as an unordered set of event times, so callers do not need to pre-sort.

## Victor-Purpura distance

`victor_purpura(a, b, *, cost)` is computed with the standard dynamic program.
Let `g[i][j]` be the minimum cost to align the first `i` spikes of `a` with the
first `j` spikes of `b`. Then

```
g[0][j] = j
g[i][0] = i
g[i][j] = min(
    g[i-1][j] + 1,                      # delete a_i
    g[i][j-1] + 1,                      # insert b_j
    g[i-1][j-1] + cost * |a_i - b_j|,   # shift a_i onto b_j
)
```

The implementation keeps only the previous row, so it runs in O(n*m) time and
O(m) memory.

## van Rossum distance

`van_rossum(a, b, *, tau)` uses the closed form of the distance between the two
trains after convolution with a causal exponential kernel `exp(-t / tau)`. The
inner product of two filtered trains is, up to the shared factor `tau / 2`,

```
S(x, y) = sum over spike pairs exp(-|xi - yj| / tau)
```

so the squared distance is

```
D^2 = 0.5 * (S(a, a) + S(b, b) - 2 * S(a, b))
```

This avoids numerical integration. A tiny negative value from floating-point
cancellation is clamped to zero before the square root.

### Normalization

With this definition the distance between an empty train and a single spike is
`sqrt(0.5)`, and as `tau` grows large the distance approaches
`|len(a) - len(b)| / sqrt(2)`. The choice is stated so results are reproducible
and comparable. Libraries that normalize the empty-versus-single distance to 1
differ from this by a constant factor.

## ISI-distance

`isi_distance(a, b, *, interval)` computes the time-averaged ISI dissimilarity
over a finite interval `[t_start, t_end]`.

### ISI function and edge convention

For a train with spikes s_1 < s_2 < ... < s_N the instantaneous ISI at time t
is the length of the inter-spike interval containing t. At the boundaries:

- **Before the first spike** (t < s_1): auxiliary ISI = `max(s_1 - t_start,
  s_2 - s_1)` if N > 1, else `s_1 - t_start`.
- **After the last spike** (t > s_N): auxiliary ISI = `max(t_end - s_N,
  s_N - s_{N-1})` if N > 1, else `t_end - s_N`.
- **Empty train**: replaced by auxiliary spikes at `[t_start, t_end]`, giving a
  constant ISI of `t_end - t_start`.

This convention matches pyspike 0.9.0 (`isi_distance_python` in
`cython/python_backend.py`) exactly.

### Algorithm

1. Validate and sort both trains; deduplicate (matching pyspike's `np.unique`).
2. Compute the initial ISIs `nu1`, `nu2` using `_initial_isi`.
3. Sweep the merged event sequence in O(n + m) time, accumulating the integral
   of `|nu1 - nu2| / max(nu1, nu2)` over each constant segment.
4. Divide by the interval length.

### Validation

All reference values were computed with pyspike 0.9.0 and embedded as golden
constants in `tests/test_isi_distance.py`. The pure-Python implementation
matches pyspike to floating-point identity (error == 0.0) on all cases:
identical trains, high-rate vs single spike, two vs four spikes, empty vs
non-empty, trains with crossing ISIs, and single-spike trains.

## SPIKE-distance

`spike_distance(a, b, *, interval)` computes the time-averaged SPIKE-distance
profile over a finite interval `[t_start, t_end]`.

### Profile definition

For each train, at any time t that lies in an ISI `(t_p, t_f)` of that train,
the local spike-time distance is the linear interpolation

```
s(t) = (dt_p * (t_f - t) + dt_f * (t - t_p)) / isi
```

where `dt_p` is the distance from the preceding spike to the nearest spike in
the other train, and `dt_f` is the same for the following spike. The nearest
spike is located via `_get_min_dist`, which searches the sorted reference train
with early exit and includes the auxiliary boundary positions.

The instantaneous SPIKE-distance is

```
S(t) = 0.5 * (s_1(t) * isi_2 + s_2(t) * isi_1) / mean_isi^2
```

where `mean_isi = 0.5 * (isi_1 + isi_2)`. The profile is piecewise-linear:
within each segment between consecutive event times `s_i(t)` is linear in t, so
`S(t)` is also linear (the segment start and end values differ in general). The
scalar distance is

```
D_S = (1 / (t_end - t_start)) * integral_{t_start}^{t_end} S(t) dt
```

computed as a trapezoidal sum over the segments.

### Edge convention

The convention matches pyspike 0.9.0 (`spike_distance_python` in
`cython/python_backend.py`) exactly.

For each train with N spikes `s_1 < ... < s_N`:

- **Auxiliary boundary positions** (used as the start/end anchors in
  `_get_min_dist` to define nearest-neighbor distances for spikes near the
  interval edges):
  - `aux_before = min(t_start, s_1 - (s_2 - s_1))` if N > 1, else `t_start`.
  - `aux_after  = max(t_end,   s_N + (s_N - s_{N-1}))` if N > 1, else `t_end`.
  These extrapolate one ISI beyond the train, so the "nearest neighbor" of a
  spike near the boundary is not artificially biased toward the edge.

- **Leading ISI** (for a train whose first spike is after `t_start`):
  `max(s_1 - t_start, s_2 - s_1)` if N > 1, or `s_1 - t_start` if N == 1.

- **Trailing ISI** (after the last spike):
  `max(t_end - s_N, s_N - s_{N-1})` if N > 1, or `t_end - s_N` if N == 1.

- **Empty train**: replaced by auxiliary spikes at `[t_start, t_end]`, matching
  pyspike's `get_spikes_non_empty`. The resulting N=2 "train" has `aux_before =
  t_start`, `aux_after = t_end`, leading ISI `t_end - t_start`.

- **Coincident spikes** (same time in both trains): the profile value is defined
  as 0.0 at the event, matching pyspike.

### Algorithm

1. Validate and sort both trains; deduplicate (matching pyspike's `np.unique`).
2. Compute auxiliary boundary positions for each train.
3. Initialize `t_p`, `t_f`, `dt_p`, `dt_f`, `isi`, and `s` for both trains at
   `t_start`.
4. Sweep the merged event sequence in O(n + m) time. At each event:
   - Compute `y_seg_end` from `_dist_at_t` (end value of the current segment).
   - Accumulate the trapezoidal area `(event_t - prev_t) * 0.5 * (y_start + y_end)`.
   - Advance indices and recompute `dt_f`, `isi`, `s` for the next segment.
5. Add the final segment from the last event to `t_end`.
6. Divide the accumulated integral by `t_end - t_start`.

### Validation

All reference values were computed with pyspike 0.9.0 and embedded as golden
constants in `tests/test_spike_distance.py`. The pure-Python implementation
matches pyspike to floating-point identity (max error 5.6e-17, one ULP) on all
cases: identical trains, single differing spike, high rate vs single, empty vs
non-empty, crossing spikes, two vs four spikes, single-spike trains, and both
empty.

## Schreiber similarity

`schreiber(a, b, *, sigma)` is the cosine similarity of the two trains after
convolution with a Gaussian of standard deviation `sigma`. The Gaussian inner
product reduces to a closed form, so with `K(d) = exp(-d^2 / (4 * sigma^2))` and
`S(x, y) = sum over spike pairs K(xi - yj)` the similarity is

```
S(a, b) / sqrt(S(a, a) * S(b, b))
```

Cauchy-Schwarz bounds this by 1; a tiny floating-point overshoot is clamped. Two
empty trains are defined as similarity 1.0 and a single empty train as 0.0.

## Hunter-Milton similarity

`hunter_milton(a, b, *, tau)` scores each spike by `exp(-dt / tau)` where `dt` is
the distance to its nearest neighbor in the other train, averages that over one
train, and then averages the two directions. Nearest neighbors are found by
binary search on the sorted target train, so the cost is O((n + m) log m). The
empty-train conventions match the Schreiber measure.

## Pairwise matrices

`pairwise(trains, metric)` evaluates a caller-supplied metric over every ordered
pair of trains and returns a list of rows. It assumes nothing about the metric,
so it makes no symmetry shortcut; the cost is O(n^2) metric calls.

## NumPy fast path (optional)

`van_rossum_numpy.van_rossum_matrix(trains, *, tau)` computes the N x N
pairwise van Rossum distance matrix using NumPy broadcasting on the cross-kernel
sums. It replaces the O(n*m) pure-Python merge sweep with a vectorized outer
difference, which is faster for moderate to large trains. Self-sums are still
computed via the O(n) pure-Python markage recursion (one per train, not per
pair). The module is only imported when NumPy is present; the rest of the
package has no dependency on it.

## Why pure Python

The core metrics are short, well-specified algorithms. Implementing them without
NumPy keeps installation trivial and the package usable anywhere. NumPy is
available as an optional extra (`pip install spikedist[fast]`), never a
hard requirement.
