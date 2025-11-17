import copy
from dataclasses import dataclass, field

import torch
from prettytable import PrettyTable
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ratcon.models import RationaleSelectorModel

from .metrics import build_eval_table, _build_label_groups
from .models import ExpertModel
from .utils import should_disable_tqdm


@dataclass
class BranchNode:
    stage: int
    path: tuple[int, ...]
    selector: RationaleSelectorModel
    expert: ExpertModel
    children: list["BranchNode"] = field(default_factory=list)


@dataclass
class BatchTensors:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    ner_tags: torch.Tensor | None = None


def prepare_selector_backbone(selector_cfg):
    base = SentenceTransformer(selector_cfg.sbert_name)
    pooler = copy.deepcopy(base[1])
    hidden_dim = base[0].auto_model.config.hidden_size
    with torch.no_grad():
        null_emb = base.encode([""], convert_to_tensor=True).squeeze(0)
    return pooler, hidden_dim, null_emb


def prepare_expert_backbone(model_cfg):
    base = SentenceTransformer(model_cfg.sbert_name)
    pooler = copy.deepcopy(base[1])
    hidden_dim = base[0].auto_model.config.hidden_size
    return pooler, hidden_dim


def clone_selector_components(pooler_template, null_embedding):
    pooler = copy.deepcopy(pooler_template)
    return pooler, null_embedding.clone()


def clone_expert_pooler(pooler_template):
    return copy.deepcopy(pooler_template)


def build_branch_tree(
    selector_cfg,
    model_cfg,
    num_stages,
    num_factors,
    device,
    selector_backbone,
    expert_backbone,
):
    selector_pooler_template, selector_hidden_dim, selector_null_embedding = selector_backbone
    expert_pooler_template, expert_hidden_dim = expert_backbone
    path_to_node: dict[tuple[int, ...], BranchNode] = {}
    leaf_name_to_path: dict[str, tuple[int, ...]] = {}

    def _build(stage: int, path: tuple[int, ...]) -> BranchNode:
        selector_pooler, selector_null = clone_selector_components(
            selector_pooler_template, selector_null_embedding
        )
        selector = RationaleSelectorModel(
            selector_cfg,
            pooler=selector_pooler,
            embedding_dim=selector_hidden_dim,
            null_embedding=selector_null,
        ).to(device)
        expert_pooler = clone_expert_pooler(expert_pooler_template)
        expert = ExpertModel(
            model_cfg,
            pooler=expert_pooler,
            embedding_dim=expert_hidden_dim,
        ).to(device)
        node = BranchNode(stage=stage, path=path, selector=selector, expert=expert, children=[])
        path_to_node[path] = node
        if stage + 1 < num_stages:
            for idx in range(num_factors):
                child_path = path + (idx,)
                child = _build(stage + 1, child_path)
                node.children.append(child)
        return node

    root = _build(0, tuple())
    nodes = list(path_to_node.values())
    return root, nodes, path_to_node, leaf_name_to_path


def prepare_batch(batch, device, *, require_labels: bool = False) -> BatchTensors:
    embeddings = batch["embeddings"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    ner_tags = batch.get("ner_tags")
    if ner_tags is None and require_labels:
        return BatchTensors(embeddings, attention_mask, None)
    if ner_tags is not None:
        ner_tags = ner_tags.to(device, non_blocking=True)
    return BatchTensors(embeddings, attention_mask, ner_tags)


def update_epoch_metrics(sums, counts, metrics):
    for name, value in metrics.items():
        sums[name] += value
        counts[name] += 1


def leaf_factor_name(leaf_name_to_path, path: tuple[int, ...], factor_idx: int) -> str:
    full_path = path + (factor_idx,)
    suffix = "_".join(str(idx) for idx in full_path) or "root"
    name = f"leaf_{suffix}"
    leaf_name_to_path.setdefault(name, full_path)
    return name


def accumulate_leaf_stats(
    leaf_name_to_path,
    num_factors,
    stats_dict,
    per_class_stats,
    routing,
    mask,
    ner_tags,
    path,
    label_groups,
):
    valid = mask > 0
    if ner_tags is None:
        return
    gold = (ner_tags > 0) & valid
    predictions = routing.argmax(dim=-1)

    for idx in range(num_factors):
        name = leaf_factor_name(leaf_name_to_path, path, idx)
        pred_idx = (predictions == idx) & valid
        tp = (pred_idx & gold).sum().item()
        fp = (pred_idx & (~gold)).sum().item()
        fn = ((~pred_idx) & gold).sum().item()

        leaf_counts = stats_dict.setdefault(name, dict(tp=0, fp=0, fn=0))
        leaf_counts["tp"] += tp
        leaf_counts["fp"] += fp
        leaf_counts["fn"] += fn

        if label_groups and per_class_stats is not None:
            label_map = per_class_stats.setdefault(
                name, {label: dict(tp=0, fp=0, fn=0) for label in label_groups.keys()}
            )
            for label, indices in label_groups.items():
                class_mask = torch.zeros_like(valid, dtype=torch.bool)
                for label_id in indices:
                    class_mask |= ner_tags == label_id
                class_mask = class_mask & valid
                tp_c = (pred_idx & class_mask).sum().item()
                fp_c = (pred_idx & (~class_mask)).sum().item()
                fn_c = ((~pred_idx) & class_mask).sum().item()
                label_counts = label_map[label]
                label_counts["tp"] += tp_c
                label_counts["fp"] += fp_c
                label_counts["fn"] += fn_c


def finalize_leaf_metrics(stats_dict, per_class_stats, label_groups):
    results = {}
    for name, counts in stats_dict.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        result = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        if per_class_stats and label_groups:
            label_metrics = {}
            for label in label_groups.keys():
                cls_counts = per_class_stats[name][label]
                tp_c, fp_c, fn_c = cls_counts["tp"], cls_counts["fp"], cls_counts["fn"]
                if (tp_c + fp_c + fn_c) == 0:
                    continue
                precision_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
                recall_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
                f1_c = (
                    2 * precision_c * recall_c / (precision_c + recall_c)
                    if (precision_c + recall_c) > 0
                    else 0.0
                )
                label_metrics[label] = {
                    "precision": precision_c,
                    "recall": recall_c,
                    "f1": f1_c,
                    "tp": tp_c,
                    "fp": fp_c,
                    "fn": fn_c,
                }
            if label_metrics:
                result["per_class"] = label_metrics
        results[name] = result
    return results


def log_f1_table(title: str, metrics_dict: dict[str, dict[str, float]], logger) -> None:
    if not metrics_dict:
        logger.info("%s: (no metrics)", title)
        return
    table = build_eval_table(metrics_dict)
    logger.info("\n%s:\n%s", title, table)


def log_leaf_ranking(
    title: str,
    metrics_dict: dict[str, dict[str, float]],
    logger,
    top_k: int | None = None,
    dev_metrics: dict[str, dict[str, float]] | None = None,
    sorted_items: list[tuple[str, dict[str, float]]] | None = None,
):
    if not metrics_dict:
        logger.info("%s ranking: (no metrics)", title)
        return
    items = sorted_items or sorted_leaves(metrics_dict)
    if top_k is not None:
        items = items[:top_k]
    table = PrettyTable()
    table.field_names = [
        "rank",
        "leaf",
        "val_f1",
        "val_precision",
        "val_recall",
        "dev_f1",
        "dev_precision",
        "dev_recall",
    ]
    for rank, (leaf, stats) in enumerate(items, start=1):
        dev_stats = (dev_metrics or {}).get(leaf)
        table.add_row(
            [
                rank,
                leaf,
                f"{stats.get('f1', 0.0):.4f}",
                f"{stats.get('precision', 0.0):.4f}",
                f"{stats.get('recall', 0.0):.4f}",
                f"{(dev_stats or {}).get('f1', float('nan')):.4f}" if dev_stats else "",
                f"{(dev_stats or {}).get('precision', float('nan')):.4f}" if dev_stats else "",
                f"{(dev_stats or {}).get('recall', float('nan')):.4f}" if dev_stats else "",
            ]
        )
    logger.info("\n%s ranking:\n%s", title, table.get_string())


def sorted_leaves(metrics_dict: dict[str, dict[str, float]]):
    return sorted(metrics_dict.items(), key=lambda item: item[1].get("f1", 0.0), reverse=True)


def best_leaf(metrics_dict: dict[str, dict[str, float]]):
    if not metrics_dict:
        return None
    return max(metrics_dict.items(), key=lambda item: item[1].get("f1", 0.0))


def evaluate_leaf_on_loader(trainer, leaf_name, loader, logger, tag: str | None = None):
    path = trainer.leaf_name_to_path.get(leaf_name)
    if path is None:
        logger.warning("Leaf %s not found; cannot run dev evaluation.", leaf_name)
        return None
    if len(path) != trainer.num_stages:
        logger.warning(
            "Leaf %s path length mismatch; expected %d got %d.",
            leaf_name,
            trainer.num_stages,
            len(path),
        )
        return None
    node_paths = [tuple(path[:i]) for i in range(trainer.num_stages)]
    nodes = []
    for node_path in node_paths:
        node = trainer.path_to_node.get(node_path)
        if node is None:
            logger.warning("Missing node for path %s; skipping dev eval for %s.", node_path, leaf_name)
            return None
        nodes.append(node)
    branch_factors = path[:-1]
    final_factor = path[-1]

    stats = dict(tp=0, fp=0, fn=0)
    label_names = loader.label_names if hasattr(loader, "label_names") else None
    label_groups = _build_label_groups(label_names)
    per_class = (
        {label: dict(tp=0, fp=0, fn=0) for label in label_groups.keys()}
        if label_groups
        else None
    )

    trainer._set_mode(train=False)
    iterator = tqdm(
        loader,
        desc=f"Leaf Eval {leaf_name}",
        disable=should_disable_tqdm(metrics_only=True),
    )
    with torch.no_grad():
        for batch in iterator:
            tensors = prepare_batch(batch, trainer.device, require_labels=True)
            if tensors.ner_tags is None:
                continue

            cur_embeddings = tensors.embeddings
            cur_mask = tensors.attention_mask

            for stage_idx, node in enumerate(nodes):
                _, _, selector_out = trainer._selector_forward(
                    node.selector, cur_embeddings, cur_mask
                )
                selection_mask = trainer._build_selection_mask(selector_out["gates"], cur_mask)
                cur_embeddings = cur_embeddings * selection_mask.unsqueeze(-1)
                cur_mask = (cur_mask * selection_mask.long()).clamp(max=1)

                _, _, expert_out = trainer._expert_forward(
                    node.expert, cur_embeddings, cur_mask
                )

                if stage_idx < len(nodes) - 1:
                    target_factor = branch_factors[stage_idx]
                    predictions = expert_out["pi"].argmax(dim=-1)
                    child_bool = (cur_mask > 0) & (predictions == target_factor)
                    if not child_bool.any():
                        cur_mask = cur_mask.new_zeros(cur_mask.shape)
                        break
                        cur_embeddings = cur_embeddings * child_bool.unsqueeze(-1).to(cur_embeddings.dtype)
                        cur_mask = child_bool.long()
                else:
                    valid = cur_mask > 0
                    if not valid.any():
                        break
                    predictions = expert_out["pi"].argmax(dim=-1)
                    pred_idx = valid & (predictions == final_factor)
                    gold = (tensors.ner_tags > 0) & valid
                    stats["tp"] += (pred_idx & gold).sum().item()
                    stats["fp"] += (pred_idx & (~gold)).sum().item()
                    stats["fn"] += ((~pred_idx) & gold).sum().item()
                    if per_class is not None:
                        for label, indices in label_groups.items():
                            class_mask = torch.zeros_like(valid, dtype=torch.bool)
                            for label_id in indices:
                                class_mask |= tensors.ner_tags == label_id
                            class_mask = class_mask & valid
                            tp_c = (pred_idx & class_mask).sum().item()
                            fp_c = (pred_idx & (~class_mask)).sum().item()
                            fn_c = ((~pred_idx) & class_mask).sum().item()
                            counts = per_class[label]
                            counts["tp"] += tp_c
                            counts["fp"] += fp_c
                            counts["fn"] += fn_c
                    break

    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    metrics = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    if per_class is not None:
        per_class_metrics = {}
        for label, counts in per_class.items():
            tp_c, fp_c, fn_c = counts["tp"], counts["fp"], counts["fn"]
            if tp_c + fp_c + fn_c == 0:
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
            metrics["per_class"] = per_class_metrics

    suffix = f" ({tag})" if tag else ""
    logger.debug(
        "Leaf %s%s dev evaluation: precision %.4f recall %.4f f1 %.4f",
        leaf_name,
        suffix,
        precision,
        recall,
        f1,
    )
    return metrics
