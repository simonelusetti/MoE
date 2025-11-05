#!/usr/bin/env python
"""
FrameNet bucket statistics.
Computes average number of concept buckets per sentence and their token sizes.
"""

import argparse
import ast
import os
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm


FRAMENET_SPLIT_FILES = {
    "train": "fn1.7/fn1.7.fulltext.train.syntaxnet.conll",
    "validation": "fn1.7/fn1.7.dev.syntaxnet.conll",
    "test": "fn1.7/fn1.7.test.syntaxnet.conll",
}

FRAME_ELEMENT_KEYWORDS = {
    "temporal": {
        "time",
        "date",
        "duration",
        "period",
        "era",
        "age",
        "season",
        "year",
        "month",
        "day",
        "hour",
        "moment",
        "frequency",
        "interval",
        "tempo",
    },
    "spatial": {
        "location",
        "place",
        "area",
        "region",
        "site",
        "path",
        "route",
        "destination",
        "source",
        "direction",
        "position",
        "ground",
        "surface",
        "setting",
        "locative",
        "boundary",
        "landmark",
    },
    "participant": {
        "agent",
        "actor",
        "participant",
        "person",
        "people",
        "individual",
        "experiencer",
        "self",
        "subject",
        "object",
        "patient",
        "theme",
        "entity",
        "group",
        "organization",
        "company",
        "customer",
        "buyer",
        "seller",
        "speaker",
        "listener",
        "addressee",
        "performer",
        "owner",
        "user",
        "audience",
    },
    "causal": {"cause", "reason", "purpose", "motivation", "stimulus", "trigger", "explanation"},
    "manner": {"manner", "means", "method", "instrument", "medium", "way", "style", "technique", "tool", "process", "vehicle", "mode"},
    "quantity": {"amount", "quantity", "degree", "extent", "number", "ratio", "percentage", "size", "measure", "magnitude", "value"},
    "communication": {"message", "topic", "content", "statement", "utterance", "information", "text", "speech", "language", "word", "phrase", "question", "answer", "report"},
    "mental": {"emotion", "feeling", "state", "attitude", "belief", "thought", "memory", "perception", "cognition", "reaction", "experience", "desire"},
    "event": {"event", "activity", "action", "process", "occurrence", "happening", "step", "phase", "task", "operation"},
    "comparison": {"comparison", "standard", "reference", "benchmark", "similarity", "difference", "contrast"},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=str, default="train", help="FrameNet split: train/validation/test.")
    parser.add_argument(
        "--subset",
        type=float,
        default=1.0,
        help="If <=1, fraction of split to use; otherwise absolute number of examples.",
    )
    parser.add_argument(
        "--framenet-cache",
        type=Path,
        default=None,
        help="Optional path to FrameNet cache root or fn1.7 subdirectory.",
    )
    return parser.parse_args()


def _locate_cache(cache_arg: Path | None) -> Path:
    if cache_arg is not None:
        return cache_arg.expanduser().resolve()
    dataset_cache = os.environ.get("HF_DATASETS_CACHE")
    if dataset_cache:
        return Path(dataset_cache).expanduser().resolve()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return (Path(hf_home) / "datasets").expanduser().resolve()
    raise FileNotFoundError("Unable to locate FrameNet cache root.")


def _normalize_split_name(split: str) -> str:
    s = split.lower()
    return "validation" if s in ("dev", "validation") else s


def _decode_token(value: str) -> str:
    if value.startswith("b'") and value.endswith("'"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, bytes):
                return parsed.decode("utf-8", errors="ignore")
        except (SyntaxError, ValueError):
            return value
    return value


def _parse_conll(path: Path):
    tokens, tags = [], []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                if tokens:
                    yield {"tokens": tokens, "frame_elements": tags}
                    tokens, tags = [], []
                continue
            parts = stripped.split("\t")
            token = _decode_token(parts[1])
            fe = parts[14] if parts[14] else "O"
            tokens.append(token)
            tags.append(fe)
    if tokens:
        yield {"tokens": tokens, "frame_elements": tags}


def _frame_element_to_bucket(label: str) -> str:
    if not label or label == "O":
        return "none"
    base = label[2:] if label.startswith(("B-", "I-")) else label
    base = base.lower()
    if not base or base == "o":
        return "none"
    for bucket, keywords in FRAME_ELEMENT_KEYWORDS.items():
        if any(keyword in base for keyword in keywords):
            return bucket
    return "other"


def main():
    args = parse_args()
    cache_root = _locate_cache(args.framenet_cache)
    split_key = _normalize_split_name(args.split)
    rel_path = Path(FRAMENET_SPLIT_FILES[split_key])

    base_paths = [
        cache_root / "liyucheng__FrameNet_v17",
        cache_root / "fn1.7",
        cache_root,
    ]
    conll_path = None
    for base in base_paths:
        candidate = (base / rel_path).expanduser().resolve()
        if candidate.exists():
            conll_path = candidate
            break
        candidate = (base / rel_path.name).expanduser().resolve()
        if candidate.exists():
            conll_path = candidate
            break

    grouped = {}
    for example in _parse_conll(conll_path):
        key = tuple(example["tokens"])
        bucket_map = grouped.setdefault(
            key,
            {"tokens": example["tokens"], "bucket_indices": defaultdict(set)},
        )["bucket_indices"]
        for idx, label in enumerate(example["frame_elements"]):
            bucket = _frame_element_to_bucket(label)
            if bucket != "none":
                bucket_map[bucket].add(idx)

    aggregated = [
        {"tokens": entry["tokens"], "bucket_indices": {bucket: sorted(indices) for bucket, indices in entry["bucket_indices"].items()}}
        for entry in grouped.values()
    ]
    aggregated.sort(key=lambda ex: " ".join(ex["tokens"]))

    if args.subset and args.subset > 0:
        total = len(aggregated)
        keep = int(total * args.subset) if args.subset <= 1.0 else int(args.subset)
        keep = max(1, min(total, keep))
        aggregated = aggregated[:keep]

    sentence_bucket_counts = []
    bucket_sizes = []
    bucket_frequency = Counter()

    for example in tqdm(aggregated, desc="Computing stats"):
        buckets = example["bucket_indices"]
        if not buckets:
            continue

        sentence_bucket_counts.append(len(buckets))
        for bucket, indices in buckets.items():
            bucket_sizes.append(len(indices))
            bucket_frequency[bucket] += 1

    num_sentences = len(sentence_bucket_counts)
    if num_sentences == 0:
        raise RuntimeError("No sentences with annotated buckets found.")

    avg_buckets = sum(sentence_bucket_counts) / num_sentences
    avg_bucket_size = sum(bucket_sizes) / len(bucket_sizes)

    print(f"Split: {args.split}")
    print(f"Sentences analysed: {num_sentences}")
    print(f"Average buckets per sentence: {avg_buckets:.4f}")
    print(f"Average tokens per bucket: {avg_bucket_size:.4f}")

    print("\nBucket frequency (top 10):")
    for bucket, freq in bucket_frequency.most_common(10):
        print(f"  {bucket:15s} {freq}")


if __name__ == "__main__":
    main()
