"""Spike-train distance metrics in pure Python with zero dependencies."""

from .van_rossum import van_rossum
from .victor_purpura import victor_purpura

__all__ = ["van_rossum", "victor_purpura"]
__version__ = "0.1.0"
