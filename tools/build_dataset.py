#!/usr/bin/env python3
"""Pre-build MoE dataset caches under ./data."""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional, Union

from dora import to_absolute_path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import build_dataset


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
    )
    dataset.save_to_disk(str(target))
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cnn", "wikiann", "conll2003", "framenet"), required=True)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset = _parse_subset(args.subset)
    cnn_field = args.cnn_field
    if args.dataset == "cnn" and cnn_field is None:
        cnn_field = "highlights"
    dataset_config = args.config
    if args.dataset == "framenet" and dataset_config is None:
        dataset_config = "fulltext"

    prepared = []
    for split in args.splits:
        path = _prepare_dataset(
            name=args.dataset,
            split=split,
            subset=subset,
            tokenizer=args.tokenizer,
            max_length=args.max_length,
            cnn_field=cnn_field,
            dataset_config=dataset_config,
            rebuild=args.rebuild,
            shuffle=args.shuffle,
        )
        prepared.append(path)

    print("\nPrepared datasets:")
    for path in prepared:
        status = "exists" if path.exists() else "missing"
        print(f"  {status:>6} {path}")


if __name__ == "__main__":
    main()
