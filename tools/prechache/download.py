#!/usr/bin/env python3
"""Download and verify Hugging Face assets for offline MoE training."""

import argparse
import os
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, AutoTokenizer

_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")


def _prepare_cache(cache_dir: str) -> dict[str, Path]:
    """Ensure cache directories exist and return a mapping of their paths."""
    root = Path(os.path.expanduser(cache_dir)).resolve()
    cache_paths = {
        "root": root,
        "datasets": root / "datasets",
        "hub": root / "hub",
        "transformers": root / "transformers",
        "modules": root / "modules",
    }
    for path in cache_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return cache_paths


def _set_env(cache: dict[str, Path], *, offline: bool) -> None:
    """Point Hugging Face environment variables at the provided cache."""
    os.environ["HF_HOME"] = str(cache["root"])
    os.environ["HF_DATASETS_CACHE"] = str(cache["datasets"])
    os.environ["TRANSFORMERS_CACHE"] = str(cache["transformers"])
    os.environ["HF_HUB_CACHE"] = str(cache["hub"])
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache["hub"])
    os.environ["HF_MODULES_CACHE"] = str(cache["modules"])
    if offline:
        for var in _OFFLINE_ENV_VARS:
            os.environ[var] = "1"
    else:
        for var in _OFFLINE_ENV_VARS:
            os.environ.pop(var, None)


def _download_conll(cache: dict[str, Path], revision: str) -> None:
    print("Caching CoNLL-2003 splits:")
    dataset = load_dataset("conll2003", revision=revision, cache_dir=str(cache["datasets"]))
    for split, ds in dataset.items():
        print(f"  {split}: {len(ds)} examples")


def _download_wikiann(cache: dict[str, Path], languages: Iterable[str]) -> None:
    print("Caching WikiANN modules and splits:")
    snapshot_download(
        "wikiann",
        repo_type="dataset",
        cache_dir=str(cache["modules"]),
        local_dir=None,
        local_dir_use_symlinks=False,
    )
    for lang in languages:
        dataset = load_dataset("wikiann", lang, cache_dir=str(cache["datasets"]))
        sizes = {split: len(ds) for split, ds in dataset.items()}
        print(f"  Language '{lang}':")
        for split, size in sizes.items():
            print(f"    {split}: {size} examples")


def _download_models(cache: dict[str, Path], model_names: Iterable[str]) -> None:
    print("Caching transformer models:")
    for name in model_names:
        print(f"  Downloading '{name}'...")
        AutoConfig.from_pretrained(name, cache_dir=str(cache["transformers"]))
        AutoTokenizer.from_pretrained(name, cache_dir=str(cache["transformers"]), use_fast=True)
        AutoModel.from_pretrained(name, cache_dir=str(cache["transformers"]))
        print(f"    Cached '{name}'")


def _verify_wikiann(cache: dict[str, Path], languages: Iterable[str]) -> bool:
    """Attempt to load the WikiANN splits offline."""
    from datasets import config as datasets_config  # imported here to pick up env overrides
    from datasets.download.download_config import DownloadConfig

    _set_env(cache, offline=True)
    datasets_config.HF_CACHE_HOME = cache["root"]
    datasets_config.HF_DATASETS_CACHE = cache["datasets"]
    datasets_config.HF_MODULES_CACHE = cache["modules"]

    print("Verifying offline WikiANN loads:")
    success = True
    for lang in languages:
        try:
            download_config = DownloadConfig(local_files_only=True, cache_dir=str(cache["datasets"]))
            dataset = load_dataset(
                "wikiann",
                lang,
                split="train",
                cache_dir=str(cache["datasets"]),
                download_config=download_config,
            )
            print(f"  [OK] wikiann/{lang}: {len(dataset)} train examples")
        except Exception as err:  # noqa: BLE001
            success = False
            print(f"  [FAIL] wikiann/{lang}: {err}")
    return success


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
        help="CoNLL-2003 dataset revision to download (default: %(default)s)",
    )
    parser.add_argument(
        "--wikiann-langs",
        nargs="+",
        default=["en"],
        help="WikiANN language codes to cache and verify (default: %(default)s)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["sentence-transformers/all-MiniLM-L6-v2"],
        help="Transformer model ids to cache (default: %(default)s)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip offline verification once downloads complete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = _prepare_cache(args.cache_dir)

    _set_env(cache, offline=False)
    _download_conll(cache, args.revision)
    _download_wikiann(cache, args.wikiann_langs)
    _download_models(cache, args.models)

    if args.no_verify:
        return

    if not _verify_wikiann(cache, args.wikiann_langs):
        raise SystemExit("Offline verification failed. See logs above for details.")


if __name__ == "__main__":
    main()
