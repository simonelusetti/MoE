#!/usr/bin/env python
"""
Partition-based similarity exploration on WikiANN.
Enumerates token partitions and compares cosine similarity against the full sentence embedding.
"""

import argparse
import math
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from tqdm import tqdm

import torch
from datasets import load_dataset
from prettytable import PrettyTable
from sentence_transformers import SentenceTransformer


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
        "--max-partitions",
        type=int,
        default=None,
        help="Optional cap on the number of partitions evaluated overall.",
    )
    parser.add_argument(
        "--partitions-per-length",
        type=int,
        default=256,
        help="Number of unique partition/complement pairs to sample per sentence length "
        "(<=0 enumerates all pairs).",
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
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_subset(split: str, subset: float | None, seed: int):
    ds = load_dataset("wikiann", "en", split=split)
    if subset and subset > 0:
        rng = random.Random(seed)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        keep = int(len(ds) * subset) if subset <= 1.0 else int(subset)
        ds = ds.select(sorted(indices[: max(1, min(len(ds), keep))]))
    return ds


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
        if target is not None and 0 < target < unique_limit:
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

    log_lines: list[str] = []

    def log_line(message: str = "") -> None:
        print(message)
        log_lines.append(message)

    dataset = load_subset(args.split, args.subset, args.seed)
    total_examples = len(dataset)
    log_line(f"Loaded {total_examples} examples from wikiann:{args.split} (device={device}).")
    if args.partitions_per_length is not None and args.partitions_per_length > 0:
        log_line(f"Sampling {args.partitions_per_length} unique partition pairs per sentence length.")
    else:
        log_line("Evaluating all possible partition pairs per sentence length (may be very slow).")

    stats = defaultdict(list)
    bucket_scores = defaultdict(list)
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

    # sweep sentences, measuring cosine response for each partition pair
    for example_idx, example in enumerate(tqdm(dataset, desc="Processing examples")):
        quota = None if quota_plan is None else quota_plan[example_idx]
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]
        if len(tokens) < args.min_length:
            continue

        full_text = " ".join(tokens)
        full_embedding = model.encode(
            full_text,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False,
        ).to(device=device, dtype=torch.float32)

        sentence_partitions = 0
        length = len(tokens)
        partitions = partition_cache.get(length)
        if partitions is None:
            partitions = sample_partition_pairs(length, rng, args.partitions_per_length)
            partition_cache[length] = partitions

        # evaluate each partition against the full-sentence embedding
        for left_indices, right_indices in partitions:
            if quota is not None and sentence_partitions >= quota:
                break

            left_text = " ".join(tokens[i] for i in left_indices)
            right_text = " ".join(tokens[i] for i in right_indices)

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

            left_entities = sum(ner_tags[i] != 0 for i in left_indices)
            right_entities = sum(ner_tags[i] != 0 for i in right_indices)

            stats[(left_entities, right_entities)].append(sim_sum)

            if left_entities > 0 and right_entities == 0:
                bucket = "left_only"
            elif right_entities > 0 and left_entities == 0:
                bucket = "right_only"
            elif left_entities > 0 and right_entities > 0:
                bucket = "mixed"
            else:
                bucket = "none"
            bucket_scores[bucket].append(sim_sum)
            sentence_partitions += 1
            processed_partitions += 1

        if sentence_partitions:
            processed_sentences += 1

    if not stats:
        raise RuntimeError("No partitions processed; dataset may be empty or filtering removed all examples.")

    summary = []
    for (ent_left, ent_right), values in stats.items():
        count = len(values)
        mean_value = sum(values) / count
        summary.append((ent_left, ent_right, count, mean_value))

    summary.sort(key=lambda item: (item[0] + item[1], item[0], item[1]))

    summary_table = PrettyTable()
    summary_table.field_names = ["entities_left", "entities_right", "count", "mean_cosine_sum"]
    for ent_left, ent_right, count, mean_value in summary:
        summary_table.add_row([ent_left, ent_right, count, f"{mean_value:.4f}"])

    log_line("")
    log_line("Partition summary (cosine left/right vs full):")
    log_line(summary_table.get_string())
    log_line("")
    log_line(f"Processed sentences: {processed_sentences}")
    log_line(f"Evaluated partitions: {processed_partitions}")

    if bucket_scores:
        bucket_table = PrettyTable()
        bucket_table.field_names = ["bucket", "count", "mean_cosine_sum"]
        for bucket, scores in sorted(bucket_scores.items()):
            mean_val = sum(scores) / len(scores)
            bucket_table.add_row([bucket, len(scores), f"{mean_val:.4f}"])
        log_line("")
        log_line("Aggregate by entity distribution:")
        log_line(bucket_table.get_string())

    if args.output:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(log_lines) + ("\n" if log_lines else ""), encoding="utf-8")
        print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
