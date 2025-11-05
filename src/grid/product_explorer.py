"""Expose the product explorer entry point under its own module.

This thin wrapper lets `dora grid product_explorer` resolve the
Explorer-decorated callable declared in `src.grid.explorer`.
"""

from src.grid.explorer import product_explorer as explorer

__all__ = ["explorer"]
