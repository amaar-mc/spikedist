from collections.abc import Sequence
from math import isfinite


def check_train(name: str, train: Sequence[float]) -> list[float]:
    """Validate a spike train and return its times as a sorted list of floats.

    Spikes are treated as an unordered set of event times, so the result is
    sorted ascending. Non-finite values are rejected with a clear error.
    """
    out: list[float] = []
    for index, value in enumerate(train):
        time = float(value)
        if not isfinite(time):
            raise ValueError(f"{name} contains a non-finite spike time at index {index}: {value!r}")
        out.append(time)
    out.sort()
    return out
