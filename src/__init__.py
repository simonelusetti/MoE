"""Expert MoE rationale project."""

import sys
from importlib import import_module

# Provide compatibility alias for Dora expecting `src.grids`.
_grid_module = import_module(f"{__name__}.grid")
sys.modules.setdefault(f"{__name__}.grids", _grid_module)
grids = _grid_module

__all__ = ["grids"]
