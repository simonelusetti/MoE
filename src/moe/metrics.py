import math
from collections import OrderedDict

import torch
from prettytable import PrettyTable
from tqdm import tqdm

from .utils import should_disable_tqdm


def filter_metric(name, value, *, model=None, weights=None):
    if value is None or not isinstance(value, (int, float)):
        return False
    if not math.isfinite(value):
        return False
    if model is not None:
        use_balance = model.use_balance if hasattr(model, "use_balance") else True
        if name == "balance" and use_balance is False:
            return False
        use_diversity = model.use_diversity if hasattr(model, "use_diversity") else True
        if name == "diversity" and use_diversity is False:
            return False
        if name == "continuity":
            if not hasattr(model, "use_continuity"):
                raise AttributeError("Model missing 'use_continuity' attribute required for continuity metrics.")
            if model.use_continuity is False:
                return False
    return True


def _normalize_label_name(name: str | None) -> str | None:
    if not name:
        return None
    if name.upper() == "O":
        return None
    if "-" in name:
        name = name.split("-", 1)[-1]
    return name.upper()


def _build_label_groups(label_names: list[str] | None) -> OrderedDict[str, list[int]]:
    groups: OrderedDict[str, list[int]] = OrderedDict()
    if not label_names:
        return groups
    for idx, raw in enumerate(label_names):
        normalized = _normalize_label_name(raw)
        if normalized is None:
            continue
        groups.setdefault(normalized, []).append(idx)
    return groups


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

    class_labels = sorted(
        {
            label
            for stats in factor_metrics.values()
            for label in (stats.get("per_class") or {}).keys()
        }
    )

    table = PrettyTable()
    table.field_names = ["factor", "precision", "recall", "f1", *class_labels]
    for factor, stats in sorted(factor_metrics.items()):
        row = [
            factor,
            f"{stats.get('precision', 0.0):.4f}",
            f"{stats.get('recall', 0.0):.4f}",
            f"{stats.get('f1', 0.0):.4f}",
        ]
        per_class = stats.get("per_class") or {}
        for label in class_labels:
            f1_value = per_class.get(label, {}).get("f1", 0.0)
            row.append(f"{f1_value:.4f}")
        table.add_row(row)
    return table.get_string()


def _require_ner_tags(batch, *, logger=None):
    if "ner_tags" in batch:
        return True
    if logger is not None:
        logger.warning("Batch missing 'ner_tags'; skipping factor evaluation.")
    return False


def evaluate_factor_metrics(model, loader, device, logger=None, threshold: float | None = None):
    model.eval()
    num_factors = model.num_experts
    stats = [dict(tp=0, fp=0, fn=0) for _ in range(num_factors)]
    has_labels = False

    label_names = loader.label_names if hasattr(loader, "label_names") else None
    label_groups = _build_label_groups(label_names)
    per_class_stats = None
    if label_groups:
        per_class_stats = [
            {label: dict(tp=0, fp=0, fn=0) for label in label_groups.keys()} for _ in range(num_factors)
        ]

    with torch.no_grad():
        iterator = tqdm(loader, desc="Factor Eval", disable=should_disable_tqdm(metrics_only=True))
        for batch in iterator:
            if not _require_ner_tags(batch, logger=logger):
                continue
            has_labels = True

            embeddings = batch["embeddings"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            ner_tags = batch["ner_tags"].to(device, non_blocking=True)
            outputs = model(embeddings, attention_mask)
            routing_scores = outputs["pi"]
            max_scores, routing = routing_scores.max(dim=-1)

            valid = attention_mask > 0
            gold = (ner_tags > 0) & valid
            strong = max_scores >= threshold if threshold is not None else None

            factor_preds = []
            for idx in range(num_factors):
                pred = (routing == idx) & valid
                if strong is not None:
                    pred = pred & strong
                factor_preds.append(pred)

            gold_by_class = {}
            if label_groups:
                for label, indices in label_groups.items():
                    class_mask = torch.zeros_like(valid, dtype=torch.bool)
                    for label_id in indices:
                        class_mask |= (ner_tags == label_id)
                    gold_by_class[label] = class_mask & valid

            for idx in range(num_factors):
                pred = factor_preds[idx]
                tp = (pred & gold).sum().item()
                fp = (pred & (~gold)).sum().item()
                fn = ((~pred) & gold).sum().item()

                stats[idx]["tp"] += tp
                stats[idx]["fp"] += fp
                stats[idx]["fn"] += fn

                if per_class_stats:
                    for label, class_mask in gold_by_class.items():
                        tp_c = (pred & class_mask).sum().item()
                        fp_c = (pred & (~class_mask)).sum().item()
                        fn_c = ((~pred) & class_mask).sum().item()

                        cls_counts = per_class_stats[idx][label]
                        cls_counts["tp"] += tp_c
                        cls_counts["fp"] += fp_c
                        cls_counts["fn"] += fn_c

    if not has_labels:
        return {}

    names = [f"factor_{idx}" for idx in range(len(stats))]
    return summarize_factor_counts(stats, per_class_stats, label_groups, names=names)


def summarize_factor_counts(stats, per_class_stats=None, label_groups=None, names=None):
    results = {}

    def _iter_items():
        if isinstance(stats, dict):
            for name, counts in stats.items():
                per_cls = None
                if isinstance(per_class_stats, dict):
                    per_cls = per_class_stats.get(name)
                yield name, counts, per_cls
        else:
            stats_list = list(stats)
            resolved_names = names or [f"factor_{idx}" for idx in range(len(stats_list))]
            for idx, counts in enumerate(stats_list):
                name = resolved_names[idx] if idx < len(resolved_names) else f"factor_{idx}"
                if isinstance(per_class_stats, dict):
                    per_cls = per_class_stats.get(name)
                elif isinstance(per_class_stats, (list, tuple)):
                    per_cls = per_class_stats[idx]
                else:
                    per_cls = None
                yield name, counts, per_cls

    label_keys = list(label_groups.keys()) if label_groups else None
    for name, counts, per_cls in _iter_items():
        tp = counts.get("tp", 0)
        fp = counts.get("fp", 0)
        fn = counts.get("fn", 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        result = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

        if per_cls:
            per_class_metrics = {}
            labels = label_keys or per_cls.keys()
            for label in labels:
                cls_counts = per_cls.get(label)
                if not cls_counts:
                    continue
                tp_c = cls_counts.get("tp", 0)
                fp_c = cls_counts.get("fp", 0)
                fn_c = cls_counts.get("fn", 0)
                if (tp_c + fp_c + fn_c) == 0:
                    continue
                precision_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
                recall_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
                f1_c = (
                    2 * precision_c * recall_c / (precision_c + recall_c)
                    if (precision_c + recall_c) > 0
                    else 0.0
                )
                per_class_metrics[label] = {
                    "precision": precision_c,
                    "recall": recall_c,
                    "f1": f1_c,
                    "tp": tp_c,
                    "fp": fp_c,
                    "fn": fn_c,
                }
            if per_class_metrics:
                result["per_class"] = per_class_metrics
        results[name] = result
    return results
