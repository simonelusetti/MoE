#!/usr/bin/env python3
"""Sequential pseudo-grid runner for the composite selector+expert model."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from _pseudo_grid_base import DEFAULT_GRID, parse_common_args, run_jobs  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_common_args(__doc__ or "", "src.train_composite", argv or sys.argv[1:])
    return run_jobs(DEFAULT_GRID, args, label_prefix="composite:")


if __name__ == "__main__":
    raise SystemExit(main())
