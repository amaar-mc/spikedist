# Charter

## Purpose

Provide correct, dependency-free implementations of the most cited spike-train
distance metrics, callable directly on plain sequences of spike times, so that
comparing spike trains does not require a heavy framework or a compiled
extension.

## Scope

- Victor-Purpura distance (edit-based).
- van Rossum distance (kernel-based).
- Clear input validation and exact, documented normalizations.
- Future: Schreiber and Hunter-Milton measures, multi-unit van Rossum, a fast
  van Rossum path, and pairwise distance matrices.

## Non-goals

- ISI-distance, SPIKE-distance, and SPIKE-synchronization. `pyspike` is the
  reference implementation for those and is not worth duplicating.
- Spike sorting, simulation, or a full neuroscience analysis framework.
- A required NumPy dependency. NumPy may appear only as an optional accelerator.

## Principles

- Correctness first. Every metric is tested against exact reference values and
  the metric axioms it satisfies.
- Zero runtime dependencies.
- Small, stable, fully typed API. Metric parameters are explicit.
- Honest documentation of each metric's definition and normalization.

## Audience

Computational neuroscience researchers and students, neuromorphic engineers, and
anyone prototyping or teaching spike-train similarity.
