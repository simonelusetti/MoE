#!/usr/bin/env python3
"""Pseudo-grid runner for stage-wise branching experiments."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from _pseudo_grid_base import GridJob, parse_common_args, run_jobs  # noqa: E402

HYPERPARAM_GRID = [
    {
        "label": "conll_lr5e-5_tau07",
        "dataset": "conll2003",
        "subset": 1.0,
        "model.optim.lr": 5e-5,
        "model.contrastive_tau": 0.07,
    },
    {
        "label": "conll_lr3e-5_tau05",
        "dataset": "conll2003",
        "subset": 1.0,
        "model.optim.lr": 3e-5,
        "model.contrastive_tau": 0.05,
    },
    {
        "label": "conll_balance_high",
        "dataset": "conll2003",
        "subset": 1.0,
        "model.loss_weights.balance": 0.03,
        "model.loss_weights.diversity": 0.02,
    },
    {
        "label": "conll_selector_strict",
        "dataset": "conll2003",
        "subset": 1.0,
        "composite.selector_threshold": 0.6,
        "model.loss_weights.overlap": 0.02,
    },
    {
        "label": "conll_more_experts",
        "dataset": "conll2003",
        "subset": 1.0,
        "model.expert.num_experts": 6,
        "model.loss_weights.diversity": 0.05,
    },
    {
        "label": "conll_stage_short",
        "dataset": "conll2003",
        "subset": 1.0,
        "composite.stage_epochs": "[3,3,3,3,3]",
        "model.optim.lr": 4e-5,
    },
]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_common_args(__doc__ or "", "src.train_composite_branching", argv or sys.argv[1:])
    return run_jobs(HYPERPARAM_GRID, args, label_prefix="branching:")


if __name__ == "__main__":
    raise SystemExit(main())
