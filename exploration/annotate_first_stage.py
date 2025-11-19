#!/usr/bin/env python3
"""Annotate first-stage selector/expert outputs starting from raw FrameNet data."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

import torch
from datasets import Dataset, load_from_disk
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_composite_branching import BranchingCompositeTrainer
from src.utils import configure_runtime


def decode_token(token: str) -> str:
    if token.startswith("b'") or token.startswith('b"'):
        try:
            literal = ast.literal_eval(token)
            if isinstance(literal, bytes):
                return literal.decode("utf-8", errors="ignore")
        except (SyntaxError, ValueError, UnicodeDecodeError):
            pass
    return token


def build_frame_entity_tags(row, *, limit: int | None = None):
    lexical_units = row.get("lexical_units", [])
    frame_elements = row.get("frame_elements", [])
    tags = []
    for lu, fe in zip(lexical_units, frame_elements):
        is_frame = bool(lu)
        is_entity = fe != "O"
        if is_frame and is_entity:
            tags.append("frame+entity")
        elif is_frame:
            tags.append("frame")
        elif is_entity:
            tags.append("entity")
        else:
            tags.append("none")
    if limit is not None:
        tags = tags[:limit]
    return tags


def load_config(args):
    if args.config:
        return OmegaConf.load(args.config)
    if not args.xp_signature:
        raise ValueError("Either --config or --xp-signature must be provided")
    config_path = Path("outputs") / "xps" / args.xp_signature / ".hydra" / "config.yaml"
    return OmegaConf.load(config_path)


def resolve_checkpoint(args):
    if args.checkpoint:
        return Path(args.checkpoint)
    if not args.xp_signature:
        raise ValueError("Either --checkpoint or --xp-signature must be provided")
    return Path("outputs") / "xps" / args.xp_signature / "branching_composite.pth"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Path to raw FrameNet dataset (load_from_disk directory)")
    parser.add_argument("--output", required=True, help="Directory to save the annotated dataset")
    parser.add_argument("--xp-signature", default=None, help="Experiment signature under outputs/xps")
    parser.add_argument("--config", default=None, help="Optional explicit config path")
    parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path")
    parser.add_argument("--device", default="cuda", help="Computation device (default: cuda if available)")
    return parser.parse_args()


def encode_tokens(tokens, tokenizer, model, device):
    if not tokens:
        return [], torch.empty(0, model.config.hidden_size), torch.zeros(0, dtype=torch.long), []

    sentence = " ".join(tokens)
    batch_encoding = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=False,
    )
    input_ids = batch_encoding["input_ids"].squeeze(0)
    subword_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    inputs = {k: v.to(device) for k, v in batch_encoding.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state.squeeze(0)
    attention = batch_encoding["attention_mask"].squeeze(0).cpu()

    return subword_tokens, hidden.cpu(), attention


def main():
    args = parse_args()
    cfg = load_config(args)
    configure_runtime(cfg)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = resolve_checkpoint(args)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.sbert_name, use_fast=True)
    encoder = AutoModel.from_pretrained(cfg.model.sbert_name).to(device)
    encoder.eval()

    trainer = BranchingCompositeTrainer(cfg, device)
    state = torch.load(checkpoint, map_location=device)
    for node in trainer.nodes:
        key = BranchingCompositeTrainer._path_key(node.path)
        node.selector.load_state_dict(state["selectors"][key])
        node.expert.load_state_dict(state["experts"][key])
    root = trainer.root

    dataset = load_from_disk(args.dataset)

    records = []
    with torch.no_grad():
        for row in tqdm(dataset, desc="Annotating"):
            raw_tokens = [decode_token(tok) for tok in row.get("tokens", [])]
            frame_tags = build_frame_entity_tags(row)

            subword_tokens, hidden, attention = encode_tokens(raw_tokens, tokenizer, encoder, device)
            seq_len = hidden.size(0)
            if seq_len == 0:
                continue

            embeddings = hidden.unsqueeze(0).to(device)
            attention_mask = attention.unsqueeze(0).to(device)

            selector_out = root.selector(embeddings, attention_mask)
            selection_mask = trainer._build_selection_mask(selector_out["gates"], attention_mask)
            selected = selection_mask.squeeze(0).bool()

            selected_embeddings = embeddings * selection_mask.unsqueeze(-1)
            selected_attention = (attention_mask * selection_mask.long()).clamp(max=1)
            expert_out = root.expert(selected_embeddings, selected_attention)
            expert_ids = expert_out["pi"].squeeze(0).argmax(dim=-1)

            selected_list = selected.tolist()
            expert_ids = expert_ids.tolist()
            expert_assignment = [int(expert_ids[i]) if selected_list[i] else None for i in range(seq_len)]

            records.append(
                {
                    "model_tokens": subword_tokens,
                    "frame_entity_tag": frame_tags,
                    "selected": selected_list,
                    "expert_assignment": expert_assignment,
                }
            )

    annotated = Dataset.from_list(records)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    annotated.save_to_disk(str(output_path))


if __name__ == "__main__":
    main()
