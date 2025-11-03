#!/usr/bin/env python
"""
Explore how cosine similarity between sentence partitions and the full sentence
varies with the number of entity tokens per partition.
"""

import argparse
import ast
import math
import os
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from tqdm import tqdm

import torch
from datasets import load_dataset
from prettytable import PrettyTable
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Partition-based similarity exploration on token-level concept datasets.")
    parser.add_argument(
        "--subset",
        type=float,
        default=1.0,
        help="If <=1, fraction of the train split to use; otherwise absolute number of examples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling when sampling a subset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to analyse (default: train).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("wikiann", "framenet"),
        default="wikiann",
        help="Dataset to analyse (default: wikiann).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer checkpoint.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=3,
        help="Minimum number of tokens required to consider a sentence.",
    )
    parser.add_argument(
        "--max-partitions",
        type=int,
        default=None,
        help="Optional cap on the number of partition pairs evaluated (for quick experiments).",
    )
    parser.add_argument(
        "--partitions-per-length",
        type=int,
        default=256,
        help=(
            "Number of unique partition/complement pairs to evaluate for each sentence length. "
            "Set to <=0 to enumerate all possible pairs (may be extremely slow)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file to write the formatted summary to.",
    )
    parser.add_argument(
        "--framenet-cache",
        type=Path,
        default=None,
        help="Optional path to the FrameNet cache containing fn1.7/*.conll files (auto-detected if omitted).",
    )
    return parser.parse_args()


def resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    "entity": 1,
    "participant": 2,
    "temporal": 3,
    "spatial": 4,
    "causal": 5,
    "manner": 6,
    "quantity": 7,
    "communication": 8,
    "mental": 9,
    "event": 10,
    "comparison": 11,
    "other": 20,
}

BUCKET_VIEW_ORDER = {"left": 0, "right": 1, "either": 2}


def _normalize_split_name(split: str) -> str:
    normalized = split.lower()
    if normalized in ("dev", "validation"):
        return "validation"
    if normalized not in FRAMENET_SPLIT_FILES:
        return normalized
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
        except (ValueError, SyntaxError):
            return value
    return value


def _parse_framenet_conll(path: Path) -> list[dict[str, list[str]]]:
    examples: list[dict[str, list[str]]] = []
    tokens: list[str] = []
    frame_elements: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                if tokens:
                    examples.append(
                        {
                            "tokens": tokens,
                            "frame_elements": frame_elements,
                        }
                    )
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
        examples.append({"tokens": tokens, "frame_elements": frame_elements})
    return examples


def _frame_element_to_bucket(label: str) -> str:
    if not label or label == "O":
        return "none"
    base = label
    if base.startswith(("B-", "I-")):
        base = base[2:]
    base = base.lower()
    if not base or base == "o":
        return "none"
    for bucket, keywords in FRAME_ELEMENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in base:
                return bucket
    return "other"


def _dominant_bucket(bucket_counts: Counter[str]) -> str:
    filtered = {bucket: count for bucket, count in bucket_counts.items() if bucket != "none" and count > 0}
    if not filtered:
        return "none"
    max_count = max(filtered.values())
    candidates = sorted(bucket for bucket, count in filtered.items() if count == max_count)
    return candidates[0]


def _bucket_order_key(bucket: str) -> tuple[int, str]:
    return (BUCKET_ORDER.get(bucket, 99), bucket)


def _locate_framenet_cache(cache_arg: Path | None) -> Path:
    if cache_arg is not None:
        cache_path = cache_arg.expanduser().resolve()
        if cache_path.is_dir():
            return cache_path
        raise FileNotFoundError(f"Provided FrameNet cache directory does not exist: {cache_path}")
    dataset_cache = os.environ.get("HF_DATASETS_CACHE")
    if dataset_cache:
        candidate_root = Path(dataset_cache) / "liyucheng__FrameNet_v17"
        if (candidate_root / "fn1.7").is_dir():
            return candidate_root / "fn1.7"
        if candidate_root.is_dir():
            return candidate_root
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidate_root = Path(hf_home) / "datasets" / "liyucheng__FrameNet_v17"
        if (candidate_root / "fn1.7").is_dir():
            return candidate_root / "fn1.7"
        if candidate_root.is_dir():
            return candidate_root
    raise FileNotFoundError(
        "Unable to locate FrameNet cache. Provide --framenet-cache or set HF_HOME/HF_DATASETS_CACHE appropriately."
    )


def _load_wikiann_examples(split: str, subset: float | None, seed: int) -> list[dict[str, list[str]]]:
    ds = load_dataset("wikiann", "en", split=split)
    indices = _select_indices(len(ds), subset, seed)
    if len(indices) != len(ds):
        ds = ds.select(indices)
    examples = []
    for record in ds:
        tokens = record.get("tokens") or []
        ner_tags = record.get("ner_tags") or []
        if len(tokens) != len(ner_tags):
            continue
        concept_tags = ["entity" if tag != 0 else "none" for tag in ner_tags]
        examples.append(
            {
                "tokens": tokens,
                "concept_tags": concept_tags,
                "raw_tags": ner_tags,
            }
        )
    return examples


def _load_framenet_examples(
    split: str,
    subset: float | None,
    seed: int,
    cache_dir: Path | None,
) -> list[dict[str, list[str]]]:
    cache_root = _locate_framenet_cache(cache_dir)
    normalized = _normalize_split_name(split)
    if normalized not in FRAMENET_SPLIT_FILES:
        raise ValueError(f"Unsupported FrameNet split: {split}")
    rel_path = Path(FRAMENET_SPLIT_FILES[normalized])
    candidate_paths = [
        cache_root / rel_path,
    ]
    if cache_root.parent != cache_root:
        candidate_paths.append(cache_root.parent / rel_path)
    if len(rel_path.parts) > 1:
        candidate_paths.append(cache_root / rel_path.name)
        candidate_paths.append(cache_root.parent / rel_path.name)
    file_path = None
    for candidate in candidate_paths:
        if candidate and candidate.exists():
            file_path = candidate
            break
    if file_path is None:
        raise FileNotFoundError(
            f"FrameNet CONLL file not found. Tried: {', '.join(str(c) for c in candidate_paths if c)}"
        )
    examples = _parse_framenet_conll(file_path)
    indices = _select_indices(len(examples), subset, seed)
    examples = [examples[idx] for idx in indices]
    for example in examples:
        frame_elements = example["frame_elements"]
        concept_tags = [_frame_element_to_bucket(tag) for tag in frame_elements]
        example["concept_tags"] = concept_tags
    return examples


def load_examples(
    dataset: str,
    split: str,
    subset: float | None,
    seed: int,
    framenet_cache: Path | None,
) -> list[dict[str, list[str]]]:
    if dataset == "wikiann":
        return _load_wikiann_examples(split, subset, seed)
    if dataset == "framenet":
        return _load_framenet_examples(split, subset, seed, framenet_cache)
    raise ValueError(f"Unsupported dataset: {dataset}")


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def canonicalize_partition(selection: tuple[int, ...], length: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not selection or len(selection) >= length:
        return tuple(), tuple()
    complement = tuple(idx for idx in range(length) if idx not in selection)
    if not complement:
        return tuple(), tuple()
    if len(selection) > len(complement) or (len(selection) == len(complement) and selection > complement):
        return complement, selection
    return selection, complement


def count_unique_partition_pairs(length: int) -> int:
    if length <= 1:
        return 0
    total = 0
    half = length // 2
    for r in range(1, half + 1):
        combinations_count = math.comb(length, r)
        if r == length - r:
            combinations_count //= 2
        total += combinations_count
    return total


def enumerate_unique_partition_pairs(length: int):
    if length <= 1:
        return []
    indices = tuple(range(length))
    unique_pairs = []
    for r in range(1, length):
        for combo in combinations(indices, r):
            left, right = canonicalize_partition(combo, length)
            if not left or not right:
                continue
            if left == combo:
                unique_pairs.append((left, right))
    return unique_pairs


def sample_partition_pairs(length: int, rng: random.Random, target: int | None):
    if length <= 1:
        return []

    unique_limit = count_unique_partition_pairs(length)
    if target is None or target <= 0 or target >= unique_limit:
        all_pairs = enumerate_unique_partition_pairs(length)
        if target is not None and target > 0 and target < unique_limit:
            return rng.sample(all_pairs, target)
        return all_pairs

    selected = set()
    samples = []
    max_attempts = max(target * 20, 200)
    attempts = 0
    while len(samples) < target and attempts < max_attempts:
        r = rng.randint(1, length - 1)
        raw = tuple(sorted(rng.sample(range(length), r)))
        left, right = canonicalize_partition(raw, length)
        attempts += 1
        if not left or not right:
            continue
        if left in selected:
            continue
        selected.add(left)
        samples.append((left, right))

    if len(samples) < target:
        all_pairs = enumerate_unique_partition_pairs(length)
        if target >= len(all_pairs):
            return all_pairs
        return rng.sample(all_pairs, target)

    return samples


def main():
    args = parse_args()
    device = resolve_device()
    model = SentenceTransformer(args.model, device=device)

    log_lines = []

    def log_line(message: str = "") -> None:
        print(message)
        log_lines.append(message)

    examples = load_examples(
        dataset=args.dataset,
        split=args.split,
        subset=args.subset,
        seed=args.seed,
        framenet_cache=args.framenet_cache,
    )
    total_examples = len(examples)
    log_line(f"Loaded {total_examples} examples from {args.dataset}:{args.split} (device={device}).")
    if args.partitions_per_length is not None and args.partitions_per_length > 0:
        log_line(f"Sampling {args.partitions_per_length} unique partition pairs per sentence length.")
    else:
        log_line("Evaluating all possible partition pairs per sentence length (may be very slow).")

    stats: dict[tuple[str, str], list[float]] = defaultdict(list)
    bucket_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    processed_sentences = 0
    processed_partitions = 0
    rng = random.Random(args.seed)
    partition_cache: dict[int, list[tuple[tuple[int, ...], tuple[int, ...]]]] = {}

    quota_plan = None
    if args.max_partitions is not None:
        max_partitions = max(0, args.max_partitions)
        if total_examples > 0:
            base, remainder = divmod(max_partitions, total_examples)
            quota_plan = [base + (1 if i < remainder else 0) for i in range(total_examples)]
        else:
            quota_plan = []

    for example_idx, example in enumerate(tqdm(examples, desc="Processing examples")):
        quota = None if quota_plan is None else quota_plan[example_idx]
        tokens = example.get("tokens") or []
        concept_tags = example.get("concept_tags") or []
        if len(tokens) < max(args.min_length, 2):
            continue
        if len(tokens) != len(concept_tags):
            continue

        full_text = " ".join(tokens)
        if not full_text.strip():
            continue

        full_embedding = model.encode(
            full_text,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False,
        )
        full_embedding = full_embedding.to(device=device, dtype=torch.float32)

        if quota is not None and quota <= 0:
            continue

        sentence_partitions = 0
        length = len(tokens)
        partitions = partition_cache.get(length)
        if partitions is None:
            partitions = sample_partition_pairs(length, rng, args.partitions_per_length)
            partition_cache[length] = partitions
        if not partitions:
            continue

        for left_indices, right_indices in partitions:
            if quota is not None and sentence_partitions >= quota:
                break

            left_tokens = [tokens[i] for i in left_indices]
            right_tokens = [tokens[i] for i in right_indices]

            left_text = " ".join(left_tokens)
            right_text = " ".join(right_tokens)

            left_emb, right_emb = model.encode(
                [left_text, right_text],
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False,
            )
            left_emb = left_emb.to(device=device, dtype=torch.float32)
            right_emb = right_emb.to(device=device, dtype=torch.float32)

            sim_left = cosine_similarity(left_emb, full_embedding)
            sim_right = cosine_similarity(right_emb, full_embedding)
            sim_sum = sim_left + sim_right

            left_bucket_counts = Counter(concept_tags[i] for i in left_indices)
            right_bucket_counts = Counter(concept_tags[i] for i in right_indices)

            dominant_left = _dominant_bucket(left_bucket_counts)
            dominant_right = _dominant_bucket(right_bucket_counts)
            stats[(dominant_left, dominant_right)].append(sim_sum)

            present_left = {bucket for bucket, count in left_bucket_counts.items() if bucket != "none" and count > 0}
            present_right = {bucket for bucket, count in right_bucket_counts.items() if bucket != "none" and count > 0}

            for bucket in present_left:
                bucket_scores[(bucket, "left")].append(sim_left)
            for bucket in present_right:
                bucket_scores[(bucket, "right")].append(sim_right)
            for bucket in (present_left | present_right):
                bucket_scores[(bucket, "either")].append(sim_sum)

            sentence_partitions += 1
            processed_partitions += 1

        if sentence_partitions > 0:
            processed_sentences += 1

    if not stats:
        message = "No partitions processed; check subset size or filtering conditions."
        log_line(message)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(message + "\n", encoding="utf-8")
        return

    summary = []
    for (dominant_left, dominant_right), values in stats.items():
        count = len(values)
        mean_value = sum(values) / count
        summary.append((dominant_left, dominant_right, count, mean_value))

    summary.sort(
        key=lambda item: (
            _bucket_order_key(item[0]),
            _bucket_order_key(item[1]),
            -item[2],
        )
    )

    summary_table = PrettyTable()
    summary_table.field_names = ["dominant_left", "dominant_right", "count", "mean_cosine_sum"]
    for dom_left, dom_right, count, mean_value in summary:
        summary_table.add_row([dom_left, dom_right, count, f"{mean_value:.4f}"])

    log_line("")
    log_line("Partition summary by dominant concept (cosine left/right vs full):")
    log_line(summary_table.get_string())
    log_line("")
    log_line(f"Processed sentences: {processed_sentences}")
    log_line(f"Evaluated partitions: {processed_partitions}")

    if bucket_scores:
        bucket_table = PrettyTable()
        bucket_table.field_names = ["bucket", "view", "count", "mean_score"]
        for (bucket, view), scores in sorted(
            bucket_scores.items(),
            key=lambda kv: (_bucket_order_key(kv[0][0]), BUCKET_VIEW_ORDER.get(kv[0][1], 99), kv[0][1]),
        ):
            mean_val = sum(scores) / len(scores)
            bucket_table.add_row([bucket, view, len(scores), f"{mean_val:.4f}"])
        log_line("")
        log_line("Aggregate by concept presence:")
        log_line(bucket_table.get_string())

    if args.output:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(log_lines) + ("\n" if log_lines else ""), encoding="utf-8")
        print(f"Saved summary to {output_path}")



if __name__ == "__main__":
    main()
