"""Spike-train distance and similarity metrics in pure Python with zero dependencies."""

import contextlib

from .hunter_milton import hunter_milton
from .isi_distance import isi_distance
from .multiunit import van_rossum_multiunit
from .pairwise import pairwise
from .schreiber import schreiber
from .van_rossum import van_rossum
from .victor_purpura import victor_purpura

__all__ = [
    "hunter_milton",
    "isi_distance",
    "pairwise",
    "schreiber",
    "van_rossum",
    "van_rossum_matrix",
    "van_rossum_multiunit",
    "victor_purpura",
]
__version__ = "0.6.0"

# ``van_rossum_matrix`` is available only when NumPy is installed
# (``pip install spikedist[fast]``). Import it lazily so the package
# still loads with zero dependencies.
with contextlib.suppress(ImportError):
    from .van_rossum_numpy import van_rossum_matrix
