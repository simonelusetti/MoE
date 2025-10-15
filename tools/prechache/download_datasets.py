#!/usr/bin/env python3
"""Download and cache datasets needed for offline training."""

import argparse
import os
from typing import Iterable

from datasets import load_dataset


def _prepare_cache(cache_dir: str) -> str:
    """Ensure cache directories exist and update HF environment variables."""
    cache_dir = os.path.expanduser(cache_dir)
    hub_cache = os.path.join(cache_dir, "hub")
    datasets_cache = os.path.join(cache_dir, "datasets")
    transformers_cache = os.path.join(cache_dir, "transformers")

    for path in (cache_dir, hub_cache, datasets_cache, transformers_cache):
        os.makedirs(path, exist_ok=True)

    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = datasets_cache
    os.environ["TRANSFORMERS_CACHE"] = transformers_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_cache
    return datasets_cache


def _ensure_online_access() -> None:
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(var, None)


def download_conll(cache_dir: str, revision: str) -> None:
    cache_path = _prepare_cache(cache_dir)
    _ensure_online_access()
    dataset = load_dataset("conll2003", revision=revision, cache_dir=cache_path)
    sizes = {split: len(ds) for split, ds in dataset.items()}
    print("Cached conll2003 splits:")
    for split, size in sizes.items():
        print(f"  {split}: {size}")


def download_wikiann(cache_dir: str, languages: Iterable[str]) -> None:
    cache_path = _prepare_cache(cache_dir)
    _ensure_online_access()
    print("Caching wikiann splits:")
    for lang in languages:
        dataset = load_dataset("wikiann", lang, cache_dir=cache_path)
        sizes = {split: len(ds) for split, ds in dataset.items()}
        print(f"  Language '{lang}':")
        for split, size in sizes.items():
            print(f"    {split}: {size}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME", os.path.expanduser("~/hf-cache")),
        help="Directory to use for the HF cache (default: %(default)s)",
    )
    parser.add_argument(
        "--revision",
        default="refs/convert/parquet",
        help="conll2003 dataset revision to download (default: %(default)s)",
    )
    parser.add_argument(
        "--wikiann-langs",
        nargs="+",
        default=["en"],
        help="WikiANN language codes to cache (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_conll(args.cache_dir, args.revision)
    download_wikiann(args.cache_dir, args.wikiann_langs)
