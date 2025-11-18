import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.train_composite_branching import BranchingCompositeTrainer  # noqa: E402


def _resolve_local_model_path(model_name: str) -> Path:
    potential_path = Path(model_name)
    if potential_path.exists():
        return potential_path
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / f"models--{model_name.replace('/', '--')}"
    snapshot_dir = model_dir / "snapshots"
    if snapshot_dir.exists():
        snapshots = sorted(snapshot_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if snapshots:
            return snapshots[0]
    raise FileNotFoundError(f"No local snapshot found for {model_name}")


def _load_vocab(model_name: str):
    snapshot = _resolve_local_model_path(model_name)
    vocab_file = snapshot / "vocab.txt"
    if not vocab_file.exists():
        raise FileNotFoundError(f"vocab.txt not found in {snapshot}")
    with vocab_file.open("r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]
    return vocab


def _decode_tokens(row, vocab, length):
    input_ids = row.get("input_ids")
    if input_ids is None:
        return [], None
    tokens = []
    words = []
    current = ""
    specials = {"[CLS]", "[SEP]", "[PAD]"}
    for idx in input_ids[:length]:
        token = vocab[idx]
        if token in specials:
            if current:
                words.append(current)
                current = ""
            continue
        tokens.append(token)
        if token.startswith("##"):
            current += token[2:]
        else:
            if current:
                words.append(current)
            current = token
    if current:
        words.append(current)
    sentence = " ".join(words) if words else None
    return tokens, sentence


def main():
    parser = argparse.ArgumentParser(description="Analyze factor assignments for sentences.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to Hydra config; defaults to outputs/xps/<sig>/.hydra/config.yaml.",
    )
    parser.add_argument(
        "--xp_signature",
        required=True,
        help="Experiment signature (expects checkpoint at outputs/xps/<sig>/expert_model.pth).",
    )
    parser.add_argument(
        "--dataset",
        default="data/wikiann_validation.pt",
        help="Path to cached dataset directory (load_from_disk).",
    )
    parser.add_argument(
        "--output",
        default="outputs/factor_analysis.jsonl",
        help="Output JSONL file to write analysis results.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional number of sentences to analyze.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path("outputs") / "xps" / args.xp_signature / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    cfg = OmegaConf.load(config_path)
    device = torch.device(args.device)

    checkpoint_path = Path("outputs") / "xps" / args.xp_signature / "branching_composite.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    local_model_path = _resolve_local_model_path(cfg.model.sbert_name)
    cfg.model.sbert_name = str(local_model_path)
    cfg.selector_model.sbert_name = str(local_model_path)

    trainer = BranchingCompositeTrainer(cfg, device)
    state = torch.load(checkpoint_path, map_location=device)
    for node in trainer.nodes:
        key = BranchingCompositeTrainer._path_key(node.path)
        node.selector.load_state_dict(state["selectors"][key])
        node.expert.load_state_dict(state["experts"][key])
    model = trainer.root.expert
    vocab = _load_vocab(cfg.model.sbert_name)

    dataset = load_from_disk(args.dataset)
    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def analyze_node(node, embeddings, mask, token_indices, tokens, records):
        with torch.no_grad():
            selector_out = node.selector(embeddings, mask)
            selection_mask = trainer._build_selection_mask(selector_out["gates"], mask)
            selected_embeddings = embeddings * selection_mask.unsqueeze(-1)
            selected_mask = (mask * selection_mask.long()).clamp(max=1)
            expert_out = node.expert(selected_embeddings, selected_mask)

        valid = selected_mask.squeeze(0).bool()
        selected_positions = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        stage_entries = []
        if selected_positions.numel() > 0:
            predictions = expert_out["pi"].squeeze(0).argmax(dim=-1)
            for pos in selected_positions.tolist():
                absolute_idx = token_indices[pos]
                token_text = tokens[absolute_idx] if tokens and absolute_idx < len(tokens) else None
                stage_entries.append(
                    {
                        "position": absolute_idx,
                        "token": token_text,
                        "expert": int(predictions[pos].item()),
                    }
                )
        records.append({"stage": node.stage, "selections": stage_entries})

        if not node.children or selected_positions.numel() == 0:
            return

        predictions = expert_out["pi"].squeeze(0).argmax(dim=-1)
        for child_idx, child in enumerate(node.children):
            child_positions = [
                pos for pos in selected_positions.tolist() if predictions[pos].item() == child_idx
            ]
            if not child_positions:
                continue
            index_tensor = torch.tensor(child_positions, device=embeddings.device, dtype=torch.long)
            child_embeddings = selected_embeddings.index_select(1, index_tensor)
            child_mask = torch.ones(1, index_tensor.numel(), dtype=torch.long, device=embeddings.device)
            child_token_indices = [token_indices[pos] for pos in child_positions]
            analyze_node(child, child_embeddings, child_mask, child_token_indices, tokens, records)

    with output_path.open("w", encoding="utf-8") as writer:
        iterator = tqdm(
            enumerate(dataset),
            total=total,
            desc="Analyzing factors",
        )
        for idx, row in iterator:
            if idx >= total:
                break

            raw_embeddings = torch.tensor(row["embeddings"], dtype=torch.float32, device=device)
            raw_attention = row.get("attention_mask")
            if raw_attention is not None:
                attention_tensor = torch.tensor(raw_attention, dtype=torch.long, device=device)
                length = int(attention_tensor.sum().item())
            else:
                length = raw_embeddings.shape[0]
            embeddings = raw_embeddings.unsqueeze(0)[:, :length, :]
            mask = torch.ones((1, length), dtype=torch.long, device=device)

            tokens, sentence = _decode_tokens(row, vocab, length)
            tokens = tokens[:length] if tokens else []
            token_indices = list(range(length))

            stage_records = []
            analyze_node(trainer.root, embeddings, mask, token_indices, tokens, stage_records)

            record = {
                "index": idx,
                "sentence": sentence,
                "tokens": tokens,
                "stages": stage_records,
            }
            writer.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
