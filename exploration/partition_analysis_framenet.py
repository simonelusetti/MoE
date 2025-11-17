#!/usr/bin/env python
"""
FrameNet factor removal analysis.
Compares cosine shift after removing all tokens of a semantic factor versus size-matched random baselines.
"""

import argparse
import ast
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

import torch
from prettytable import PrettyTable
from sentence_transformers import SentenceTransformer


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
    "causal": {
        "cause",
        "reason",
        "purpose",
        "motivation",
        "stimulus",
        "trigger",
        "explanation",
    },
    "manner": {
        "manner",
        "means",
        "method",
        "instrument",
        "medium",
        "way",
        "style",
        "technique",
        "tool",
        "process",
        "vehicle",
        "mode",
    },
    "quantity": {
        "amount",
        "quantity",
        "degree",
        "extent",
        "number",
        "ratio",
        "percentage",
        "size",
        "measure",
        "magnitude",
        "value",
    },
    "communication": {
        "message",
        "topic",
        "content",
        "statement",
        "utterance",
        "information",
        "text",
        "speech",
        "language",
        "word",
        "phrase",
        "question",
        "answer",
        "report",
    },
    "mental": {
        "emotion",
        "feeling",
        "state",
        "attitude",
        "belief",
        "thought",
        "memory",
        "perception",
        "cognition",
        "reaction",
        "experience",
        "desire",
    },
    "event": {
        "event",
        "activity",
        "action",
        "process",
        "occurrence",
        "happening",
        "step",
        "phase",
        "task",
        "operation",
    },
    "comparison": {
        "comparison",
        "standard",
        "reference",
        "benchmark",
        "similarity",
        "difference",
        "contrast",
    },
}

BUCKET_ORDER = {
    "none": 0,
    "participant": 1,
    "temporal": 2,
    "spatial": 3,
    "event": 4,
    "causal": 5,
    "manner": 6,
    "quantity": 7,
    "communication": 8,
    "mental": 9,
    "comparison": 10,
    "other": 20,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        type=float,
        default=1.0,
        help="If <=1, fraction of the split to use; otherwise absolute number of examples.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to analyse.")
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer checkpoint.",
    )
    parser.add_argument("--min-length", type=int, default=3, help="Minimum token length for a sentence.")
    parser.add_argument(
        "--framenet-cache",
        type=Path,
        default=None,
        help="Optional path to the FrameNet cache (root directory or fn1.7 subdirectory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file to write the formatted summary to.",
    )
    parser.add_argument(
        "--buckets",
        type=str,
        default=None,
        help="Optional comma-separated list of buckets to analyse (others are ignored).",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=5,
        help="Number of sentences to print with detailed factor/removal examples (default: 5).",
    )
    return parser.parse_args()


def resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _normalize_split_name(split: str) -> str:
    normalized = split.lower()
    if normalized in ("dev", "validation"):
        return "validation"
    return normalized


def _select_indices(size: int, subset: float | None, seed: int) -> list[int]:
    if subset is None or subset <= 0:
        return list(range(size))
    rng = random.Random(seed)
    indices = list(range(size))
    rng.shuffle(indices)
    if subset <= 1.0:
        keep = max(1, int(size * subset))
    else:
        keep = min(size, int(subset))
    return sorted(indices[:keep])


def _decode_conll_token(value: str) -> str:
    if value.startswith("b'") and value.endswith("'"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, bytes):
                return parsed.decode("utf-8", errors="ignore")
        except (SyntaxError, ValueError):
            return value
    return value


def _parse_framenet_conll(path: Path) -> list[dict]:
    grouped = {}
    tokens: list[str] = []
    frame_elements: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                if tokens:
                    key = tuple(tokens)
                    bucket_map = grouped.setdefault(
                        key,
                        {"tokens": tokens, "bucket_indices": defaultdict(set)},
                    )["bucket_indices"]
                    for idx, label in enumerate(frame_elements):
                        bucket = _frame_element_to_bucket(label)
                        if bucket != "none":
                            bucket_map[bucket].add(idx)
                    tokens = []
                    frame_elements = []
                continue
            parts = stripped.split("\t")
            if len(parts) < 15:
                continue
            token = _decode_conll_token(parts[1])
            fe = parts[14] if parts[14] else "O"
            tokens.append(token)
            frame_elements.append(fe)
    if tokens:
        key = tuple(tokens)
        bucket_map = grouped.setdefault(
            key,
            {"tokens": tokens, "bucket_indices": defaultdict(set)},
        )["bucket_indices"]
        for idx, label in enumerate(frame_elements):
            bucket = _frame_element_to_bucket(label)
            if bucket != "none":
                bucket_map[bucket].add(idx)

    aggregated = [
        {"tokens": entry["tokens"], "bucket_indices": {bucket: sorted(indices) for bucket, indices in entry["bucket_indices"].items()}}
        for entry in grouped.values()
    ]
    aggregated.sort(key=lambda ex: " ".join(ex["tokens"]))
    return aggregated


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


def _bucket_order_key(bucket: str) -> tuple[int, str]:
    return (BUCKET_ORDER.get(bucket, 99), bucket)


def _locate_framenet_cache(cache_arg: Path | None) -> Path:
    if cache_arg is not None:
        return cache_arg.expanduser().resolve()
    dataset_cache = os.environ.get("HF_DATASETS_CACHE")
    if dataset_cache:
        return (Path(dataset_cache) / "liyucheng__FrameNet_v17").resolve()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return (Path(hf_home) / "datasets" / "liyucheng__FrameNet_v17").resolve()
    raise FileNotFoundError("Unable to locate FrameNet cache.")


def _encode_text(model: SentenceTransformer, device: str, text: str) -> torch.Tensor:
    embedding = model.encode(
        text,
        convert_to_tensor=True,
        device=device,
        show_progress_bar=False,
    )
    if embedding.dim() > 1:
        embedding = embedding.squeeze(0)
    return embedding.to(device=device, dtype=torch.float32)


def _cosine(model: SentenceTransformer, device: str, full_embedding: torch.Tensor, text: str) -> float:
    embedding = _encode_text(model, device, text)
    return torch.nn.functional.cosine_similarity(embedding, full_embedding, dim=0).item()


def _remove_indices(tokens: list[str], indices: list[int]) -> list[str]:
    remove_set = set(indices)
    return [token for idx, token in enumerate(tokens) if idx not in remove_set]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def main():
    args = parse_args()
    device = resolve_device()
    model = SentenceTransformer(args.model, device=device)

    bucket_filter = None
    if args.buckets:
        bucket_filter = {bucket.strip().lower() for bucket in args.buckets.split(",") if bucket.strip()}

    log_lines: list[str] = []

    def log_line(message: str = "") -> None:
        print(message)
        log_lines.append(message)

    cache_root = _locate_framenet_cache(args.framenet_cache)
    split_key = _normalize_split_name(args.split)
    conll_path = (cache_root / FRAMENET_SPLIT_FILES[split_key]).resolve()
    if not conll_path.exists():
        conll_path = (cache_root / "fn1.7" / Path(FRAMENET_SPLIT_FILES[split_key]).name).resolve()

    examples = _parse_framenet_conll(conll_path)
    indices = _select_indices(len(examples), args.subset, args.seed)
    examples = [examples[idx] for idx in indices]
    total_examples = len(examples)
    log_line(f"Loaded {total_examples} examples from {conll_path.name} (device={device}).")

    rng = random.Random(args.seed)
    results_shift: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"factor": [], "random_tokens": []})
    processed_sentences = 0
    sample_remaining = args.sample_count

    for example in tqdm(examples, desc="Processing FrameNet examples"):
        tokens = example["tokens"]
        bucket_indices = example["bucket_indices"]
        if bucket_filter is not None:
            bucket_indices = {bucket: indices for bucket, indices in bucket_indices.items() if bucket in bucket_filter}
        # keep only buckets that actually have token indices
        bucket_indices = {bucket: indices for bucket, indices in bucket_indices.items() if indices}

        if len(tokens) < max(args.min_length, 2) or len(bucket_indices) < 2:
            continue

        selected_bucket = rng.choice(list(bucket_indices.keys()))
        factor_indices = bucket_indices[selected_bucket]
        k = len(factor_indices)
        full_text = " ".join(tokens)
        if not full_text.strip():
            continue

        remaining_tokens = _remove_indices(tokens, factor_indices)
        removed_tokens = [tokens[i] for i in sorted(factor_indices)]
        kept_text = " ".join(remaining_tokens)
        removed_text = " ".join(removed_tokens)
        full_embedding = _encode_text(model, device, full_text)
        cos_kept = _cosine(model, device, full_embedding, kept_text)
        cos_removed = _cosine(model, device, full_embedding, removed_text) if removed_text.strip() else 0.0
        factor_shift = (1.0 - cos_kept) / max(k, 1)
        results_shift[selected_bucket]["factor"].append(factor_shift)

        random_tokens_removed = []
        cos_rand_kept = cos_rand_removed = rand_shift = None

        available_indices = sorted({idx for indices in bucket_indices.values() for idx in indices})
        # baseline: remove same number of tokens chosen at random from factor-bearing tokens
        if len(available_indices) >= k:
            random_indices = rng.sample(available_indices, k)
            random_tokens = _remove_indices(tokens, random_indices)
            rand_removed_tokens = [tokens[i] for i in sorted(random_indices)]
            random_text = " ".join(random_tokens)
            rand_removed_text = " ".join(rand_removed_tokens)
            cos_rand_kept = _cosine(model, device, full_embedding, random_text)
            cos_rand_removed = _cosine(model, device, full_embedding, rand_removed_text) if rand_removed_text.strip() else 0.0
            rand_shift = (1.0 - cos_rand_kept) / max(k, 1)
            results_shift[selected_bucket]["random_tokens"].append(rand_shift)
            random_tokens_removed = rand_removed_tokens
        processed_sentences += 1

        if sample_remaining > 0:
            sample_remaining -= 1
            log_line("")
            log_line(f"Sample sentence #{args.sample_count - sample_remaining}:")
            log_line(f"Sentence: {full_text}")
            for bucket, indices in sorted(bucket_indices.items(), key=lambda item: _bucket_order_key(item[0])):
                removed_tokens = " | ".join(tokens[i] for i in indices)
                log_line(f"  Factor '{bucket}': {removed_tokens}")
            log_line(f"  Removed factor '{selected_bucket}': {' | '.join(tokens[i] for i in factor_indices)}")
            log_line(
                f"    cosine kept={cos_kept:.4f}, removed={cos_removed:.4f}, shift={factor_shift:.4f}"
            )
            if random_tokens_removed:
                log_line(f"  Random tokens removal: {' | '.join(random_tokens_removed)}")
                log_line(
                    f"    cosine kept={cos_rand_kept:.4f}, removed={cos_rand_removed:.4f}, shift={rand_shift:.4f}"
                )

    if processed_sentences == 0:
        raise RuntimeError("No sentences processed.")

    log_line(f"Processed sentences: {processed_sentences} / {total_examples}")

    table = PrettyTable()
    table.field_names = [
        "factor",
        "n_factor",
        "shift_factor",
        "shift_random",
        "Δ shift",
    ]

    all_factor_shifts = []
    all_random_tokens_shifts = []
    buckets_sorted = sorted(results_shift.keys(), key=_bucket_order_key)

    def fmt(value: float | None) -> str:
        return f"{value:.4f}" if value is not None else "-"

    for bucket in buckets_sorted:
        shift_data = results_shift[bucket]

        factor_shift_vals = shift_data["factor"]
        rand_tokens_shift_vals = shift_data["random_tokens"]

        shift_factor_mean = _mean(factor_shift_vals)
        shift_rand_tokens_mean = _mean(rand_tokens_shift_vals)

        if shift_factor_mean is not None:
            all_factor_shifts.extend(factor_shift_vals)
        if shift_rand_tokens_mean is not None:
            all_random_tokens_shifts.extend(rand_tokens_shift_vals)

        table.add_row(
            [
                bucket,
                len(factor_shift_vals),
                fmt(shift_factor_mean),
                fmt(shift_rand_tokens_mean),
                fmt(
                    shift_factor_mean - shift_rand_tokens_mean
                    if shift_factor_mean is not None and shift_rand_tokens_mean is not None
                    else None
                ),
            ]
        )

    # report per-factor and aggregate cosine shifts
    log_line("")
    log_line("Cosine shift metrics by factor removal vs random token baseline:")
    log_line(table.get_string())

    def fmt_overall(name: str, values: list[float]) -> str:
        return f"{name}: {sum(values) / len(values):.4f}" if values else f"{name}: -"

    log_line("")
    log_line("Overall mean shifts:")
    log_line(fmt_overall("factor removal", all_factor_shifts))
    log_line(fmt_overall("random tokens", all_random_tokens_shifts))
    if all_factor_shifts and all_random_tokens_shifts:
        log_line(
            f"Δ shift factor - random tokens: "
            f"{sum(all_factor_shifts) / len(all_factor_shifts) - sum(all_random_tokens_shifts) / len(all_random_tokens_shifts):.4f}"
        )

    if args.output:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(log_lines) + ("\n" if log_lines else ""), encoding="utf-8")
        print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
