import math
import os

import torch
from prettytable import PrettyTable
from tqdm import tqdm

from ratcon.utils import (
    build_label_groups,
    build_label_masks,
    compute_counts,
    finalize_metrics_from_counts,
    get_logger,
    merge_count_dict,
    should_disable_tqdm,
)


def configure_runtime(cfg):
    runtime = cfg.runtime
    num_threads = runtime.num_threads
    if not num_threads:
        return
    try:
        num_threads = int(num_threads)
    except (TypeError, ValueError):
        return
    if num_threads <= 0:
        return
    value = str(num_threads)
    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_var] = value
    try:
        import torch

        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(max(1, num_threads // 2))
    except Exception:
        pass


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
    stats = [dict() for _ in range(num_factors)]
    has_labels = False

    label_names = loader.label_names if hasattr(loader, "label_names") else None
    label_groups = build_label_groups(label_names)

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
            label_masks = build_label_masks(ner_tags, label_groups, valid)
            strong = max_scores >= threshold if threshold is not None else None

            factor_preds = []
            for idx in range(num_factors):
                pred = (routing == idx) & valid
                if strong is not None:
                    pred = pred & strong
                factor_preds.append(pred)

            for idx in range(num_factors):
                counts = compute_counts(factor_preds[idx], gold, label_masks=label_masks)
                merge_count_dict(stats[idx], counts)

    if not has_labels:
        return {}

    counts_map = {f"factor_{idx}": stats[idx] for idx in range(len(stats))}
    return finalize_metrics_from_counts(counts_map, label_groups)


def summarize_factor_counts(stats, per_class_stats=None, label_groups=None, names=None):
    counts_map = {}

    def _attach_counts(name, counts, per_cls):
        combined = counts.copy()
        if per_cls:
            combined = combined.copy()
            combined["per_class"] = per_cls
        counts_map[name] = combined

    if isinstance(stats, dict):
        for name, counts in stats.items():
            per_cls = None
            if isinstance(per_class_stats, dict):
                per_cls = per_class_stats.get(name)
            _attach_counts(name, counts, per_cls)
    else:
        stats_list = list(stats)
        resolved_names = names or [f"factor_{idx}" for idx in range(len(stats_list))]
        for idx, counts in enumerate(stats_list):
            name = resolved_names[idx] if idx < len(resolved_names) else f"factor_{idx}"
            if isinstance(per_class_stats, dict):
                per_cls = per_class_stats.get(name)
            elif isinstance(per_class_stats, (list, tuple)) and idx < len(per_class_stats):
                per_cls = per_class_stats[idx]
            else:
                per_cls = None
            _attach_counts(name, counts, per_cls)

    return finalize_metrics_from_counts(counts_map, label_groups)
