#!/usr/bin/env python3
"""Stage-wise branching evaluation sweep + summary table."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import torch
from omegaconf import OmegaConf
from prettytable import PrettyTable

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RATCON_ROOT = REPO_ROOT.parent / "RatCon"
if RATCON_ROOT.exists() and str(RATCON_ROOT) not in sys.path:
    sys.path.append(str(RATCON_ROOT))

from _pseudo_grid_base import GridJob  # noqa: E402
from pseudo_grid_branching import HYPERPARAM_GRID  # noqa: E402
from src.branching_tree import evaluate_leaf_on_loader, sorted_leaves  # noqa: E402
from src.data import initialize_dataloaders  # noqa: E402
from src.train_composite_branching import BranchingCompositeTrainer  # noqa: E402
from src.utils import configure_runtime, get_logger  # noqa: E402


def _load_jobs(only: set[str] | None) -> list[GridJob]:
    jobs = [GridJob.from_dict(entry) for entry in HYPERPARAM_GRID]
    if only:
        jobs = [job for job in jobs if job.label in only]
        missing = only - {job.label for job in jobs}
        if missing:
            raise SystemExit(f"Unknown grid labels requested: {', '.join(sorted(missing))}")
    return jobs


def _load_existing_overrides(xps_root: Path) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    if not xps_root.exists():
        return mapping
    for xp_dir in xps_root.iterdir():
        argv_file = xp_dir / ".argv.json"
        if not argv_file.exists():
            continue
        try:
            overrides = json.loads(argv_file.read_text())
        except json.JSONDecodeError:
            continue
        mapping[xp_dir.name] = sorted(overrides)
    return mapping


def _match_xp_dir(job_overrides: list[str], existing: dict[str, list[str]], xp_root: Path) -> Path | None:
    sorted_overrides = sorted(job_overrides)
    for sig, stored in existing.items():
        if stored == sorted_overrides:
            return xp_root / sig
    return None


def _build_cfg(xp_dir: Path):
    cfg_path = xp_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing Hydra config for {xp_dir}")
    cfg = OmegaConf.load(cfg_path)
    return cfg


def _load_checkpoint(trainer: BranchingCompositeTrainer, ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=trainer.device)
    trainer._build_tree_to_depth(trainer.num_stages)
    trainer._set_active_stage(trainer.num_stages - 1)
    trainer._load_state_from_dict(state)


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--only", nargs="*", default=None, metavar="LABEL", help="Subset of grid labels (default: all)")
    parser.add_argument("--xp-root", default="outputs/xps", help="Directory containing xp runs (default: %(default)s)")
    parser.add_argument(
        "--output", default=None, help="Optional path to dump summary as JSON/CSV (extension inferred)"
    )
    parser.add_argument("--extra", nargs="*", default=[], metavar="OVERRIDE", help="Extra overrides appended before matching")
    args = parser.parse_args(argv or sys.argv[1:])

    only = set(args.only) if args.only else None
    jobs = _load_jobs(only)
    xp_root = Path(args.xp_root)
    existing = _load_existing_overrides(xp_root)
    if not existing:
        raise SystemExit(f"No runs found under {xp_root}")

    rows = []
    for job in jobs:
        overrides = job.build_overrides() + list(args.extra)
        xp_dir = _match_xp_dir(overrides, existing, xp_root)
        if xp_dir is None:
            print(f"[skip] No xp found for {job.label}")
            continue
        print(f"[eval] {job.label} -> {xp_dir.name}")
        cfg = _build_cfg(xp_dir)
        cfg.eval.eval_only = True
        configure_runtime(cfg)
        if cfg.device == "cuda" and not torch.cuda.is_available():
            print("  CUDA unavailable; using CPU")
            cfg.device = "cpu"
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        cfg.device = device.type

        log_path = xp_dir / "eval_only.log"
        logger = get_logger(logfile=log_path)

        _train_dl, eval_dl, dev_dl = initialize_dataloaders(cfg, logger)
        if eval_dl is None:
            print("  No eval loader; skipping")
            continue
        trainer = BranchingCompositeTrainer(cfg, device)
        _load_checkpoint(trainer, xp_dir / "branching_composite.pth")
        results = trainer.evaluate(eval_dl, logger, log_to_xp=False)
        leaf_metrics = results.get("leaf_metrics", {})
        ranked = sorted_leaves(leaf_metrics)
        if not ranked:
            print("  No leaf metrics; skipping")
            continue
        best_val_leaf, best_val_stats = ranked[0]

        dev_stats = {}
        if dev_dl is not None:
            metrics = evaluate_leaf_on_loader(trainer, best_val_leaf, dev_dl, logger, tag="eval_grid")
            if metrics:
                dev_stats = metrics

        rows.append(
            {
                "label": job.label,
                "signature": xp_dir.name,
                "dataset": job.dataset,
                "subset": job.subset,
                "best_val_leaf": best_val_leaf,
                "val_f1": best_val_stats.get("f1"),
                "val_precision": best_val_stats.get("precision"),
                "val_recall": best_val_stats.get("recall"),
                "dev_f1": dev_stats.get("f1"),
                "dev_precision": dev_stats.get("precision"),
                "dev_recall": dev_stats.get("recall"),
            }
        )

    if not rows:
        print("No evaluation rows collected.")
        return 1

    table = PrettyTable()
    table.field_names = [
        "label",
        "signature",
        "dataset",
        "best_val_leaf",
        "val_f1",
        "val_precision",
        "val_recall",
        "dev_f1",
        "dev_precision",
        "dev_recall",
    ]
    for row in rows:
        table.add_row(
            [
                row["label"],
                row["signature"],
                row["dataset"],
                row["best_val_leaf"],
                _format_float(row["val_f1"]),
                _format_float(row["val_precision"]),
                _format_float(row["val_recall"]),
                _format_float(row["dev_f1"]),
                _format_float(row["dev_precision"]),
                _format_float(row["dev_recall"]),
            ]
        )
    print("\nEvaluation summary:\n" + table.get_string())

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".json":
            output_path.write_text(json.dumps(rows, indent=2))
            print(f"Saved summary to {output_path}")
        else:
            # CSV fallback
            import csv

            with output_path.open("w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            print(f"Saved summary to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
