"""Shared helpers for sequential pseudo-grid runners."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Iterable, Sequence

DEFAULT_GRID = [
    {"label": "wikiann", "dataset": "wikiann", "subset": 1.0},
    {"label": "conll2003", "dataset": "conll2003", "subset": 1.0},
    {"label": "wnut", "dataset": "wnut", "subset": 1.0},
    {"label": "ontonotes", "dataset": "ontonotes", "config": "english_v4", "subset": 1.0},
    {"label": "bc2gm", "dataset": "bc2gm", "subset": 1.0},
]


@dataclass
class GridJob:
    label: str
    dataset: str
    subset: float | None = None
    config: str | None = None
    extra_overrides: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(payload: dict) -> "GridJob":
        payload = dict(payload)
        label = payload.pop("label", payload.get("dataset"))
        dataset = payload.pop("dataset")
        subset = payload.pop("subset", None)
        config = payload.pop("config", None)
        extra = [f"{key}={value}" for key, value in payload.items()]
        return GridJob(label=label, dataset=dataset, subset=subset, config=config, extra_overrides=extra)

    def build_overrides(self) -> list[str]:
        overrides = [
            f"data.train.dataset={self.dataset}",
            f"data.eval.dataset={self.dataset}",
        ]
        if self.subset is not None:
            overrides.append(f"data.train.subset={self.subset}")
            overrides.append(f"data.eval.subset={self.subset}")
        if self.config is not None:
            overrides.append(f"data.train.config={self.config}")
            overrides.append(f"data.eval.config={self.config}")
        overrides.extend(self.extra_overrides)
        return overrides


def parse_common_args(description: str, default_train_module: str, argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--only", nargs="*", default=None, metavar="LABEL", help="Subset of grid labels to run (default: all).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use (default: current interpreter).")
    parser.add_argument("--train-module", default=default_train_module, help="Module passed to -m (default: %(default)s).")
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        metavar="OVERRIDE",
        help="Additional Hydra overrides applied to every run (e.g., train.epochs=5).",
    )
    return parser.parse_args(argv or [])


def _select_jobs(default_grid: Iterable[dict], only: set[str] | None) -> list[GridJob]:
    jobs = [GridJob.from_dict(entry) for entry in default_grid]
    if not only:
        return jobs
    filtered = [job for job in jobs if job.label in only]
    missing = only - {job.label for job in filtered}
    if missing:
        raise SystemExit(f"Unknown grid labels requested: {', '.join(sorted(missing))}")
    return filtered


def run_jobs(default_grid: Iterable[dict], args: argparse.Namespace, *, label_prefix: str = "") -> int:
    only = set(args.only) if args.only else None
    jobs = _select_jobs(default_grid, only)
    results: list[tuple[GridJob, int]] = []

    for job in jobs:
        overrides = job.build_overrides() + list(args.extra)
        cmd = [args.python, "-m", args.train_module, *overrides]
        display_label = f"{label_prefix}{job.label}" if label_prefix else job.label
        print(f"\n=== [{display_label}] running: {' '.join(cmd)}")
        if args.dry_run:
            results.append((job, 0))
            continue
        completed = subprocess.run(cmd)
        results.append((job, completed.returncode))
        if completed.returncode != 0:
            print(f"!!! [{display_label}] failed with exit code {completed.returncode}")

    print("\nSummary:")
    for job, code in results:
        display_label = f"{label_prefix}{job.label}" if label_prefix else job.label
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"  - {display_label}: {status}")

    failed = [code for _, code in results if code != 0]
    return 0 if not failed else 1
