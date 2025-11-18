#!/usr/bin/env python3
"""
Sequential “pseudo-grid” runner for trying multiple datasets locally.

The real Dora grid/explorer setup expects a Slurm-style array launcher,
but this helper simply iterates over a curated list of dataset overrides
and launches `python -m src.train ...` one run at a time.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from _pseudo_grid_base import DEFAULT_GRID, parse_common_args, run_jobs  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_common_args(__doc__ or "", "src.train", argv or sys.argv[1:])
    return run_jobs(DEFAULT_GRID, args)


if __name__ == "__main__":
    raise SystemExit(main())
