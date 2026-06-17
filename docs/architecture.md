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

## Why pure Python

The two metrics are short, well-specified algorithms. Implementing them without
NumPy keeps installation trivial and the package usable anywhere. A NumPy fast
path for large pairwise computations is planned as an optional extra, never a
hard requirement.
