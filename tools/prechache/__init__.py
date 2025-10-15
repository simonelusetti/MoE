"""Dataset caching helpers for RatCon sweeps."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

__all__ = [
    "cache_datasets",
    "parse_args",
    "main",
]

def repo_root():
    """Return the absolute path to the repository root."""

    return Path(__file__).resolve().parent.parent


def ensure_repo_on_path():
    """Ensure imports resolve relative to the repo root and set cwd."""

    root = repo_root()
    if Path.cwd() != root:
        os.chdir(root)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
def _normalize_subset(subset):
    if subset is None:
        return None
    if subset <= 0:
        raise ValueError("subset must be positive when provided")
    return subset
def cache_datasets(*, subset=1.0, rebuild=True,
                   cnn_splits=None,
                   wiki_splits=None,
                   include=None):
    """Materialise dataset caches used by RatCon grid sweeps."""

    from ratcon.data import get_dataset  # import after path setup

    datasets = set(include or {"cnn", "wikiann"})
    subset_value = _normalize_subset(subset)
    cnn_splits = list(cnn_splits or ("train", "validation", "test"))
    wiki_splits = list(wiki_splits or ("train", "validation", "test"))

    if "cnn" in datasets:
        for split in cnn_splits:
            split_subset = subset_value if split == "train" else None
            print(f"Building CNN {split} subset={split_subset!r}…", flush=True)
            get_dataset(name="cnn", split=split, subset=split_subset,
                        rebuild=rebuild)

    if "wikiann" in datasets:
        for split in wiki_splits:
            print(f"Building WikiANN {split}…", flush=True)
            get_dataset(name="wikiann", split=split, rebuild=rebuild)

    print("Dataset caching complete.", flush=True)
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=float, default=1.0,
                        help="Fraction (<=1) or absolute count of CNN train examples to keep (default: full train set).")
    parser.add_argument("--cnn-splits", nargs="+",
                        help="Specific CNN splits to cache (default: train validation test).")
    parser.add_argument("--wiki-splits", nargs="+",
                        help="Specific WikiANN splits to cache (default: train validation test).")
    parser.add_argument("--skip-cnn", action="store_true",
                        help="Skip caching all CNN splits.")
    parser.add_argument("--skip-wiki", action="store_true",
                        help="Skip caching all WikiANN splits.")
    parser.add_argument("--no-rebuild", action="store_false", dest="rebuild",
                        help="Reuse existing caches when present instead of rebuilding.")
    parser.add_argument("--rerun-grid", action="store_true",
                        help="After caching, run 'dora grid <name> --clear' with auto-confirmation.")
    parser.add_argument("--grid-name", default="grid",
                        help="Grid name to restart when --rerun-grid is used (default: grid).")
    parser.set_defaults(rebuild=True)
    return parser.parse_args(argv)
def main(argv=None):
    args = parse_args(argv)
    ensure_repo_on_path()

    targets = set()
    if not args.skip_cnn:
        targets.add("cnn")
    if not args.skip_wiki:
        targets.add("wikiann")

    cache_datasets(subset=args.subset,
                   rebuild=args.rebuild,
                   cnn_splits=args.cnn_splits,
                   wiki_splits=args.wiki_splits,
                   include=targets)

    if args.rerun_grid:
        print(f"Restarting Dora grid '{args.grid_name}' (auto-confirm)…", flush=True)
        subprocess.run(
            ["dora", "grid", args.grid_name, "--clear"],
            check=True,
            text=True,
            input="y\n",
        )
