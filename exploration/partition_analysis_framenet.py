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
    return parser.parse_args()


def resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
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


def _compute_shift(model: SentenceTransformer, device: str, full_embedding: torch.Tensor, text: str) -> float:
    edited_embedding = _encode_text(model, device, text or " ")
    cosine = torch.nn.functional.cosine_similarity(edited_embedding, full_embedding, dim=0).item()
    return 1.0 - cosine


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
    results: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"factor": [], "random_tokens": [], "random_factor": []})
    processed_sentences = 0

    for example in tqdm(examples, desc="Processing FrameNet examples"):
        tokens = example["tokens"]
        bucket_indices = example["bucket_indices"]

        if len(tokens) < max(args.min_length, 2) or not bucket_indices:
            continue

        selected_bucket = rng.choice(list(bucket_indices.keys()))
        factor_indices = bucket_indices[selected_bucket]
        full_text = " ".join(tokens)
        if not full_text.strip():
            continue

        remaining_tokens = _remove_indices(tokens, factor_indices)
        full_embedding = _encode_text(model, device, full_text)
        factor_text = " ".join(remaining_tokens)
        factor_shift = _compute_shift(model, device, full_embedding, factor_text)
        results[selected_bucket]["factor"].append(factor_shift)

        available_indices = sorted({idx for indices in bucket_indices.values() for idx in indices})
        k = len(factor_indices)
        # baseline: remove same number of tokens chosen at random from factor-bearing tokens
        if len(available_indices) >= k:
            random_indices = rng.sample(available_indices, k)
            random_tokens = _remove_indices(tokens, random_indices)
            random_text = " ".join(random_tokens)
            random_shift = _compute_shift(model, device, full_embedding, random_text)
            if random_shift is not None:
                results[selected_bucket]["random_tokens"].append(random_shift)
        # baseline: remove same number of tokens drawn from a different factor
        other_buckets = [bucket for bucket, idxs in bucket_indices.items() if bucket != selected_bucket and len(idxs) >= k]
        if other_buckets:
            other_bucket = rng.choice(other_buckets)
            other_indices = rng.sample(bucket_indices[other_bucket], k)
            other_tokens = _remove_indices(tokens, other_indices)
            other_text = " ".join(other_tokens)
            other_shift = _compute_shift(model, device, full_embedding, other_text)
            if other_shift is not None:
                results[selected_bucket]["random_factor"].append(other_shift)

        processed_sentences += 1

    if processed_sentences == 0:
        raise RuntimeError("No sentences processed.")

    log_line(f"Processed sentences: {processed_sentences} / {total_examples}")

    table = PrettyTable()
    table.field_names = [
        "factor",
        "n_factor",
        "mean_shift_factor",
        "mean_shift_rand_tokens",
        "Δ (factor-rand)",
        "mean_shift_rand_factor",
        "Δ (factor-other)",
    ]

    all_factor_shifts = []
    all_random_tokens_shifts = []
    all_random_factor_shifts = []

    for bucket in sorted(results.keys(), key=_bucket_order_key):
        data = results[bucket]
        factor_values = data["factor"]
        rand_tokens_values = data["random_tokens"]
        rand_factor_values = data["random_factor"]

        factor_mean = _mean(factor_values)
        rand_tokens_mean = _mean(rand_tokens_values)
        rand_factor_mean = _mean(rand_factor_values)

        if factor_mean is not None:
            all_factor_shifts.extend(factor_values)
        if rand_tokens_mean is not None:
            all_random_tokens_shifts.extend(rand_tokens_values)
        if rand_factor_mean is not None:
            all_random_factor_shifts.extend(rand_factor_values)

        def fmt(value: float | None) -> str:
            return f"{value:.4f}" if value is not None else "-"

        delta_tokens = (
            factor_mean - rand_tokens_mean if factor_mean is not None and rand_tokens_mean is not None else None
        )
        delta_factor = (
            factor_mean - rand_factor_mean if factor_mean is not None and rand_factor_mean is not None else None
        )

        table.add_row(
            [
                bucket,
                len(factor_values),
                fmt(factor_mean),
                fmt(rand_tokens_mean),
                fmt(delta_tokens),
                fmt(rand_factor_mean),
                fmt(delta_factor),
            ]
        )

    # report per-factor and aggregate cosine shifts
    log_line("")
    log_line("Cosine shift (1 - cosine similarity) by factor removal vs baselines:")
    log_line(table.get_string())

    def fmt_overall(name: str, values: list[float]) -> str:
        return f"{name}: {sum(values) / len(values):.4f}" if values else f"{name}: -"

    log_line("")
    log_line("Overall means:")
    log_line(fmt_overall("factor removal", all_factor_shifts))
    log_line(fmt_overall("random tokens", all_random_tokens_shifts))
    log_line(fmt_overall("random factor", all_random_factor_shifts))

    if all_factor_shifts and all_random_tokens_shifts:
        log_line(
            f"Δ factor - random tokens: "
            f"{sum(all_factor_shifts) / len(all_factor_shifts) - sum(all_random_tokens_shifts) / len(all_random_tokens_shifts):.4f}"
        )
    if all_factor_shifts and all_random_factor_shifts:
        log_line(
            f"Δ factor - random factor: "
            f"{sum(all_factor_shifts) / len(all_factor_shifts) - sum(all_random_factor_shifts) / len(all_random_factor_shifts):.4f}"
        )

    if args.output:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(log_lines) + ("\n" if log_lines else ""), encoding="utf-8")
        print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
