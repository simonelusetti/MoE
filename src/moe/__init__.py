"""Expert MoE rationale project."""

import sys
from importlib import import_module

__all__ = []

# Provide compatibility alias for Dora expecting `src.grids`.
try:
    _grid_module = import_module(".grid", __name__)
except ModuleNotFoundError:
    _grid_module = None
else:
    sys.modules.setdefault(f"{__name__}.grids", _grid_module)
    grids = _grid_module
    __all__ = ["grids"]
