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

## Why pure Python

The two metrics are short, well-specified algorithms. Implementing them without
NumPy keeps installation trivial and the package usable anywhere. A NumPy fast
path for large pairwise computations is planned as an optional extra, never a
hard requirement.
