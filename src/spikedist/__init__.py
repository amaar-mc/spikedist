"""Spike-train distance and similarity metrics in pure Python with zero dependencies."""

from .hunter_milton import hunter_milton
from .pairwise import pairwise
from .schreiber import schreiber
from .van_rossum import van_rossum
from .victor_purpura import victor_purpura

__all__ = ["hunter_milton", "pairwise", "schreiber", "van_rossum", "victor_purpura"]
__version__ = "0.2.0"
