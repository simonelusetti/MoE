#!/usr/bin/env python3
"""Pre-build MoE dataset caches under ./data."""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional, Union

from dora import to_absolute_path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import build_dataset

KNOWN_DATASETS = {
    "cnn": {"config": None, "raw_dir": "cnn_dailymail_3.0.0"},
    "wikiann": {"config": "en", "raw_dir": "wikiann_en"},
    "conll2003": {"config": None, "raw_dir": "conll2003"},
    "wnut": {"config": None, "raw_dir": "wnut"},
    "ontonotes": {"config": "english_v4", "raw_dir": "ontonotes5_english_v4"},
    "bc2gm": {"config": None, "raw_dir": "spyysalo_bc2gm_corpus"},
    "framenet": {"config": "fulltext", "raw_dir": "liyucheng_FrameNet_v17_fulltext"},
}


def _sanitize_fragment(fragment: str) -> str:
    return fragment.replace("/", "-")


def _dataset_filename(
    name: str,
    split: str,
    subset: Optional[Union[int, float]],
    cnn_field: Optional[str],
    dataset_config: Optional[str],
) -> str:
    parts = [name]
    if dataset_config:
        parts.append(_sanitize_fragment(dataset_config))
    if cnn_field:
        parts.append(cnn_field)
    parts.append(split)
    if subset is not None and subset != 1.0:
        parts.append(str(subset))
    return "_".join(parts) + ".pt"


def _parse_subset(value: Optional[str]) -> Optional[Union[int, float]]:
    if value is None:
        return None
    lowered = value.lower()
    if lowered == "none":
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"Invalid subset value: {value}") from err


def _prepare_dataset(
    *,
    name: str,
    split: str,
    subset: Optional[Union[int, float]],
    tokenizer: str,
    max_length: int,
    cnn_field: Optional[str],
    dataset_config: Optional[str],
    rebuild: bool,
    shuffle: bool,
    raw_root: Optional[str],
) -> Path:
    relative = Path("data") / _dataset_filename(name, split, subset, cnn_field, dataset_config)
    target = Path(to_absolute_path(str(relative)))

    if target.exists():
        if not rebuild:
            print(f"[skip] {target} already exists")
            return target
        print(f"[rebuild] {target}")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    else:
        print(f"[build] {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    dataset, _ = build_dataset(
        name=name,
        split=split,
        tokenizer_name=tokenizer,
        max_length=max_length,
        subset=subset,
        shuffle=shuffle,
        cnn_field=cnn_field,
        dataset_config=dataset_config,
        raw_dataset_root=raw_root,
    )
    dataset.save_to_disk(str(target))
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset",
        choices=tuple(KNOWN_DATASETS.keys()),
    )
    group.add_argument(
        "--all-known",
        action="store_true",
        help="Build caches for every dataset in KNOWN_DATASETS.",
    )
    parser.add_argument("--splits", nargs="+", default=["train"], help="Dataset splits to prepare (default: %(default)s)")
    parser.add_argument("--subset", default=None, help="Subset value matching src.data expectations (e.g. 0.1, 1000, None)")
    parser.add_argument("--tokenizer", default="sentence-transformers/all-MiniLM-L6-v2", help="Tokenizer/model id to use")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length (default: %(default)s)")
    parser.add_argument("--cnn-field", default=None, help="Field to extract from CNN DailyMail (default inferred in code)")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional dataset configuration (e.g. FrameNet 'fulltext').",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before saving")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding even if cache exists")
    parser.add_argument(
        "--raw-root",
        default=None,
        help="Optional directory containing raw splits (downloaded via download_dataset.py).",
    )
    return parser.parse_args()


def _resolve_raw_root(base_path, raw_dir, splits, name):
    base = Path(base_path)
    first_split = splits[0]
    candidates = [base]
    if raw_dir:
        candidates.append(base / raw_dir)
    for candidate in candidates:
        if (candidate / first_split).exists():
            return str(candidate)
    raise FileNotFoundError(
        f"Raw dataset splits for '{name}' not found under {base}. "
        f"Ensure you downloaded them (e.g., data/raw/{raw_dir}/{first_split})."
    )


def main() -> None:
    args = parse_args()
    subset = _parse_subset(args.subset)
    targets = []
    if args.all_known:
        for name, entry in KNOWN_DATASETS.items():
            targets.append((name, entry))
    else:
        entry = KNOWN_DATASETS.get(args.dataset, {"config": args.config, "raw_dir": None})
        if args.config is not None:
            entry = dict(entry)
            entry["config"] = args.config
        targets.append((args.dataset, entry))

    prepared = []
    for name, entry in targets:
        cnn_field = args.cnn_field
        dataset_config = entry.get("config")
        if name == "cnn" and cnn_field is None:
            cnn_field = "highlights"
        if name == "framenet" and dataset_config is None:
            dataset_config = "fulltext"
        if name == "ontonotes" and dataset_config is None:
            dataset_config = "english_v4"
        dataset_raw_root = args.raw_root
        if dataset_raw_root is not None:
            try:
                dataset_raw_root = _resolve_raw_root(dataset_raw_root, entry.get("raw_dir"), args.splits, name)
            except FileNotFoundError as err:
                print(f"[skip] {err}")
                continue
        for split in args.splits:
            path = _prepare_dataset(
                name=name,
                split=split,
            subset=subset,
            tokenizer=args.tokenizer,
            max_length=args.max_length,
            cnn_field=cnn_field,
            dataset_config=dataset_config,
            rebuild=args.rebuild,
            shuffle=args.shuffle,
                raw_root=dataset_raw_root,
        )
            prepared.append(path)

    print("\nPrepared datasets:")
    for path in prepared:
        status = "exists" if path.exists() else "missing"
        print(f"  {status:>6} {path}")


if __name__ == "__main__":
    main()
