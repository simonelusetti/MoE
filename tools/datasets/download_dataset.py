#!/usr/bin/env python3
"""Download raw Hugging Face datasets (all splits) for offline processing."""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from datasets import Dataset, DatasetDict, DatasetInfo, load_dataset

WNUT_URLS = {
    "train": "https://raw.githubusercontent.com/leondz/emerging_entities_17/master/wnut17train.conll",
    "validation": "https://raw.githubusercontent.com/leondz/emerging_entities_17/master/emerging.dev.conll",
    "test": "https://raw.githubusercontent.com/leondz/emerging_entities_17/master/emerging.test.annotated",
}

WNUT_LABEL_NAMES = [
    "O",
    "B-corporation",
    "I-corporation",
    "B-creative-work",
    "I-creative-work",
    "B-group",
    "I-group",
    "B-location",
    "I-location",
    "B-person",
    "I-person",
    "B-product",
    "I-product",
]
WNUT_LABEL_TO_ID = {label: idx for idx, label in enumerate(WNUT_LABEL_NAMES)}
KNOWN_DATASETS = {
    "cnn": {"dataset": "cnn_dailymail", "config": "3.0.0"},
    "wikiann": {"dataset": "wikiann", "config": "en"},
    "conll2003": {"dataset": "conll2003", "config": None},
    "wnut": {"dataset": None, "config": None},  # local parser only
    "ontonotes": {"dataset": "ontonotes5", "config": "english_v4"},
    "bc2gm": {"dataset": "spyysalo/bc2gm_corpus", "config": None},
    "framenet": {"dataset": "liyucheng/FrameNet_v17", "config": "fulltext"},
}

def _normalize_wnut_split(split: str) -> str:
    split = split.lower().strip().rstrip(".")
    mapping = {
        "train": "train",
        "validation": "validation",
        "val": "validation",
        "dev": "validation",
        "test": "test",
    }
    if split not in mapping:
        raise ValueError(f"Unsupported WNUT split '{split}'.")
    return mapping[split]


def _download_wnut_file(split: str, download_dir: Path) -> Path:
    normalized = _normalize_wnut_split(split)
    url = WNUT_URLS.get(normalized)
    if url is None:
        raise ValueError(f"Split '{split}' not available for WNUT dataset.")
    download_dir.mkdir(parents=True, exist_ok=True)
    filename = os.path.basename(urlparse(url).path)
    target = download_dir / filename
    if target.exists():
        return target
    try:
        with urlopen(url) as src, open(target, "wb") as dst:
            dst.write(src.read())
    except URLError as err:
        raise RuntimeError(f"Failed to download WNUT data from {url}: {err}") from err
    return target


def _parse_wnut_file(path: Path):
    sentences = []
    tokens = []
    labels = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": labels})
                    tokens = []
                    labels = []
                continue
            parts = stripped.split("\t")
            if len(parts) != 2:
                continue
            token, label = parts
            tokens.append(token)
            if label not in WNUT_LABEL_TO_ID:
                raise ValueError(f"Unknown WNUT label '{label}' in {path}")
            labels.append(WNUT_LABEL_TO_ID[label])
    if tokens:
        sentences.append({"tokens": tokens, "ner_tags": labels})
    for idx, item in enumerate(sentences):
        item["id"] = str(idx)
    return sentences


def _load_wnut_dataset(split: str, cache_dir: Path):
    path = _download_wnut_file(split, cache_dir)
    examples = _parse_wnut_file(path)
    return Dataset.from_list(examples)


SPECIAL_LOADERS = {
    "wnut": _load_wnut_dataset,
}

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset",
        help="Hugging Face dataset identifier (e.g., wikiann, ontonotes5).",
    )
    group.add_argument(
        "--all-known",
        action="store_true",
        help="Download every dataset defined in KNOWN_DATASETS.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional configuration name (e.g., en, english_v4). Ignored when --all-known is used.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="List of splits to download (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw"),
        help="Destination root directory (default: %(default)s).",
    )
    return parser.parse_args()


def save_metadata(dataset: Dataset, output_dir: Path) -> None:
    info = dataset.info
    payload = {
        "builder_name": info.builder_name,
        "config": info.config_name,
        "features": dataset.features.to_dict(),
        "citation": info.citation,
        "description": info.description,
        "homepage": info.homepage,
        "license": info.license,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_info.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _download_one(dataset_id, config, splits, output_root, label):
    if dataset_id is None:
        loader = SPECIAL_LOADERS.get(label)
        if loader is None:
            print(f"[skip] {label}: no download handler available.")
            return None
        destination = output_root / label
        destination.mkdir(parents=True, exist_ok=True)
        metadata_written = False
        for split in splits:
            print(f"[build] {label} split='{split}' via local parser")
            ds = loader(split, destination)
            if not metadata_written:
                save_metadata(ds, destination)
                metadata_written = True
            split_dir = destination / split
            print(f"[save] -> {split_dir}")
            ds.save_to_disk(str(split_dir))
        return destination
    destination = output_root / f"{dataset_id.replace('/', '_')}{f'_{config}' if config else ''}"
    destination.mkdir(parents=True, exist_ok=True)
    metadata_written = False
    for split in splits:
        print(f"[download] {dataset_id} ({config or 'default'}) split='{split}'")
        try:
            ds = load_dataset(dataset_id, config, split=split)
        except Exception as err:
            print(f"[error] Failed to download {dataset_id}:{config} split='{split}': {err}")
            return None
        if not metadata_written:
            save_metadata(ds, destination)
            metadata_written = True
        split_dir = destination / split
        print(f"[save] -> {split_dir}")
        ds.save_to_disk(str(split_dir))
    return destination


def main():
    args = parse_args()
    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    targets = []
    if args.all_known:
        for name, meta in KNOWN_DATASETS.items():
            targets.append((meta["dataset"], meta.get("config"), name))
    else:
        dataset_id = args.dataset
        config = args.config
        if dataset_id in SPECIAL_LOADERS:
            dataset_id = None
        targets.append((dataset_id, config, args.dataset))

    for dataset_id, config, label in targets:
        target_dir = _download_one(dataset_id, config, args.splits, output_root, label)
        if target_dir is None:
            print(f"[skip] {label}: requires local loader, nothing downloaded.")
            continue
        print(f"[done] Saved {label} to {target_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
