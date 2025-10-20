#!/usr/bin/env python
"""
Explore how cosine similarity between sentence partitions and the full sentence
varies with the number of entity tokens per partition.
"""

import argparse
import math
import random
from collections import defaultdict
from itertools import combinations
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional

from tqdm import tqdm

import torch
from datasets import load_dataset
from prettytable import PrettyTable
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Partition-based similarity exploration on WikiANN.")
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


def load_subset(split: str, subset: float, seed: int):
    ds = load_dataset("wikiann", "en", split=split)
    size = len(ds)
    if subset is None:
        return ds
    if subset <= 0:
        return ds

    random_generator = random.Random(seed)
    indices = list(range(size))
    random_generator.shuffle(indices)
    if subset <= 1.0:
        keep = max(1, int(size * subset))
    else:
        keep = min(size, int(subset))
    selected = indices[:keep]
    return ds.select(selected)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    args = parse_args()
    device = resolve_device()
    model = SentenceTransformer(args.model, device=device)

    log_lines = []

    def log_line(message: str = "") -> None:
        print(message)
        log_lines.append(message)

    dataset = load_subset(args.split, args.subset, args.seed)
    log_line(f"Loaded {len(dataset)} examples from wikiann:{args.split} (device={device}).")

    stats = defaultdict(list)
    bucket_scores = defaultdict(list)
    processed_sentences = 0
    processed_partitions = 0

    for example in tqdm(dataset, desc="Processing examples"):
        tokens = example.get("tokens") or []
        ner_tags = example.get("ner_tags") or []
        if len(tokens) < max(args.min_length, 2):
            continue
        if len(tokens) != len(ner_tags):
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

        sentence_partitions = 0
        indices = list(range(len(tokens)))
        for r in range(1, len(tokens)):
            for combo in combinations(indices, r):
                if len(combo) == len(tokens):
                    continue
                complement = [idx for idx in indices if idx not in combo]
                if not complement:
                    continue

                left_tokens = [tokens[i] for i in combo]
                right_tokens = [tokens[i] for i in complement]

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

                left_entities = sum(ner_tags[i] != 0 for i in combo)
                right_entities = sum(ner_tags[i] != 0 for i in complement)

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

                if args.max_partitions is not None and processed_partitions >= args.max_partitions:
                    break
            if args.max_partitions is not None and processed_partitions >= args.max_partitions:
                break

        if sentence_partitions > 0:
            processed_sentences += 1
        if args.max_partitions is not None and processed_partitions >= args.max_partitions:
            break

    if not stats:
        message = "No partitions processed; check subset size or filtering conditions."
        log_line(message)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(message + "\n", encoding="utf-8")
        return

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
