"""
SMOPCA: Spatial Multi-Omics Probabilistic Component Analysis

A GPU-accelerated implementation for analyzing spatial multi-omics data.
"""

from .model import SMOPCA
from . import utils

__version__ = "0.1.0"
__all__ = ["SMOPCA", "utils"]