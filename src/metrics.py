import math

import torch
from prettytable import PrettyTable
from tqdm import tqdm

from .utils import should_disable_tqdm


def filter_metric(name, value, *, model=None, weights=None):
    if value is None or not isinstance(value, (int, float)):
        return False
    if not math.isfinite(value):
        return False
    if weights is not None:
        weight = weights.get(name)
        if weight is not None and abs(weight) < 1e-12:
            return False
    if model is not None:
        if name == "balance" and getattr(model, "use_balance", True) is False:
            return False
        if name == "diversity" and getattr(model, "use_diversity", True) is False:
            return False
    return True


def build_train_table(train_metrics, *, model=None, weights=None):
    if not train_metrics:
        return "(no train metrics)"
    filtered = [(name, value) for name, value in sorted(train_metrics.items()) if filter_metric(name, value, model=model, weights=weights)]
    if not filtered:
        return "(no train metrics)"
    table = PrettyTable()
    table.field_names = [name for name, _ in filtered]
    table.add_row([f"{value:.4f}" for _, value in filtered])
    return table.get_string()


def build_eval_table(factor_metrics):
    if not factor_metrics:
        return "(no factor metrics)"
    table = PrettyTable()
    table.field_names = ["factor", "precision", "recall", "f1"]
    for factor, stats in sorted(factor_metrics.items()):
        table.add_row([
            factor,
            f"{stats.get('precision', 0.0):.4f}",
            f"{stats.get('recall', 0.0):.4f}",
            f"{stats.get('f1', 0.0):.4f}",
        ])
    return table.get_string()
def _require_ner_tags(batch, *, logger=None):
    if "ner_tags" in batch:
        return True
    if logger is not None:
        logger.warning("Batch missing 'ner_tags'; skipping factor evaluation.")
    return False


def evaluate_factor_metrics(model, loader, device, logger=None):
    model.eval()
    num_factors = model.num_experts
    stats = [dict(tp=0, fp=0, fn=0) for _ in range(num_factors)]
    has_labels = False

    with torch.no_grad():
        iterator = tqdm(loader, desc="Factor Eval", disable=should_disable_tqdm(metrics_only=True))
        for batch in iterator:
            if not _require_ner_tags(batch, logger=logger):
                continue
            has_labels = True

            embeddings = batch["embeddings"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            ner_tags = batch["ner_tags"].to(device, non_blocking=True)
            incoming = batch.get("incoming")
            outgoing = batch.get("outgoing")
            if incoming is not None:
                incoming = incoming.to(device, non_blocking=True)
            if outgoing is not None:
                outgoing = outgoing.to(device, non_blocking=True)

            outputs = model(embeddings, attention_mask, incoming, outgoing)
            routing = outputs["pi"].argmax(dim=-1)

            valid = attention_mask > 0
            gold = (ner_tags > 0) & valid

            for idx in range(num_factors):
                pred = (routing == idx) & valid
                tp = (pred & gold).sum().item()
                fp = (pred & (~gold)).sum().item()
                fn = ((~pred) & gold).sum().item()

                stats[idx]["tp"] += tp
                stats[idx]["fp"] += fp
                stats[idx]["fn"] += fn

    if not has_labels:
        return {}

    results = {}
    for idx, counts in enumerate(stats):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        results[f"factor_{idx}"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return results
