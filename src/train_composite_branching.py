import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from prettytable import PrettyTable
from tqdm import tqdm

from dora import get_xp, hydra_main

_CURRENT_DIR = Path(__file__).resolve().parent
_RATCON_ROOT = _CURRENT_DIR.parent.parent / "RatCon"
if _RATCON_ROOT.exists() and str(_RATCON_ROOT) not in sys.path:
    sys.path.append(str(_RATCON_ROOT))

from ratcon.losses import complement_loss, sparsity_loss as rat_sparsity_loss, total_variation_1d
from ratcon.models import RationaleSelectorModel, nt_xent

from .data import initialize_dataloaders
from .metrics import build_eval_table, _build_label_groups
from .models import ExpertModel
from .utils import configure_runtime, get_logger, should_disable_tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class BranchNode:
    stage: int
    path: tuple[int, ...]
    selector: RationaleSelectorModel
    expert: ExpertModel
    children: list["BranchNode"] = field(default_factory=list)


class BranchingCompositeTrainer:
    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.selector_cfg = cfg.selector_model
        self.model_cfg = cfg.model
        self.num_stages = int(cfg.composite.num_stages)
        if self.num_stages < 1:
            raise ValueError("composite.num_stages must be >= 1")
        self.num_factors = int(self.model_cfg.expert.num_experts)
        self.selector_weight = float(cfg.composite.loss_weights.selector)
        self.expert_weight = float(cfg.composite.loss_weights.expert)
        self.selector_threshold = float(cfg.composite.selector_threshold)
        self.grad_clip = float(cfg.train.grad_clip)

        self.selector_tau = float(self.selector_cfg.loss.tau)
        self.selector_l_comp = float(self.selector_cfg.loss.l_comp)
        self.selector_l_s = float(self.selector_cfg.loss.l_s)
        self.selector_l_tv = float(self.selector_cfg.loss.l_tv)
        self.selector_use_null = bool(self.selector_cfg.loss.use_null_target)

        self.contrastive_tau = float(self.model_cfg.contrastive_tau)

        self.path_to_node: dict[tuple[int, ...], BranchNode] = {}
        self.leaf_name_to_path: dict[str, tuple[int, ...]] = {}
        self.root = self._build_tree(stage=0, path=tuple())
        self.nodes = list(self.path_to_node.values())
        self.leaf_dev_metrics: dict[str, dict[str, float]] = {}
        self.dev_eval_top_k = 10

        selector_params = [param for node in self.nodes for param in node.selector.parameters()]
        expert_params = [param for node in self.nodes for param in node.expert.parameters()]
        self.selector_params = selector_params
        self.expert_params = expert_params

        sel_optim_cfg = self.selector_cfg.optim
        self.selector_optimizer = torch.optim.AdamW(
            selector_params,
            lr=float(sel_optim_cfg.lr),
            weight_decay=float(sel_optim_cfg.weight_decay),
            betas=tuple(sel_optim_cfg.betas),
        )

        exp_optim_cfg = self.model_cfg.optim
        self.expert_optimizer = torch.optim.AdamW(
            expert_params,
            lr=exp_optim_cfg.lr,
            weight_decay=exp_optim_cfg.weight_decay,
            betas=exp_optim_cfg.betas,
        )

    def _build_tree(self, stage: int, path: tuple[int, ...]) -> BranchNode:
        selector = RationaleSelectorModel(self.selector_cfg).to(self.device)
        expert = ExpertModel(self.model_cfg).to(self.device)
        node = BranchNode(stage=stage, path=path, selector=selector, expert=expert, children=[])
        self.path_to_node[path] = node
        if stage + 1 < self.num_stages:
            for idx in range(self.num_factors):
                child_path = path + (idx,)
                child = self._build_tree(stage + 1, child_path)
                node.children.append(child)
        return node

    def train(self, train_dl, eval_dl, dev_dl, logger, xp):
        disable_progress = should_disable_tqdm()
        best_eval = float("inf")

        for epoch in range(self.cfg.train.epochs):
            selector_sums = defaultdict(float)
            selector_counts = defaultdict(int)
            expert_sums = defaultdict(float)
            expert_counts = defaultdict(int)

            self._set_mode(train=True)

            iterator = tqdm(train_dl, desc=f"Branch Composite Train {epoch + 1}", disable=disable_progress)
            for batch in iterator:
                embeddings = batch["embeddings"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                incoming = batch.get("incoming")
                outgoing = batch.get("outgoing")
                if incoming is not None:
                    incoming = incoming.to(self.device, non_blocking=True)
                if outgoing is not None:
                    outgoing = outgoing.to(self.device, non_blocking=True)

                selector_loss, expert_loss = self._forward_batch(
                    embeddings,
                    attention_mask,
                    incoming,
                    outgoing,
                    selector_sums,
                    selector_counts,
                    expert_sums,
                    expert_counts,
                    training=True,
                )

                total_loss = self.selector_weight * selector_loss + self.expert_weight * expert_loss

                self.selector_optimizer.zero_grad(set_to_none=True)
                self.expert_optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.selector_params, self.grad_clip)
                    torch.nn.utils.clip_grad_norm_(self.expert_params, self.grad_clip)
                self.selector_optimizer.step()
                self.expert_optimizer.step()

            selector_avg = {k: selector_sums[k] / max(selector_counts[k], 1) for k in selector_sums}
            expert_avg = {k: expert_sums[k] / max(expert_counts[k], 1) for k in expert_sums}

            self._log_metrics_table(f"Selector Train (epoch {epoch + 1})", selector_avg, logger)
            self._log_metrics_table(f"Expert Train (epoch {epoch + 1})", expert_avg, logger)
            xp.link.push_metrics({f"branch/train_selector/{epoch + 1}": selector_avg})
            xp.link.push_metrics({f"branch/train_expert/{epoch + 1}": expert_avg})

            if eval_dl is None:
                continue

            eval_results = self.evaluate(eval_dl, logger, log_to_xp=False)
            selector_eval = eval_results["selector_avg"]
            expert_eval = eval_results["expert_avg"]
            leaf_metrics = eval_results["leaf_metrics"]
            sorted_leaves = self._sorted_leaves(leaf_metrics)

            self._log_metrics_table(f"Selector Eval (epoch {epoch + 1})", selector_eval, logger)
            self._log_metrics_table(f"Expert Eval (epoch {epoch + 1})", expert_eval, logger)
            self._log_f1_table(f"Leaf Factors Eval (epoch {epoch + 1})", leaf_metrics, logger)

            xp.link.push_metrics({f"branch/eval_selector/{epoch + 1}": selector_eval})
            xp.link.push_metrics({f"branch/eval_expert/{epoch + 1}": expert_eval})
            best_eval_leaf = self._best_leaf(leaf_metrics)
            if best_eval_leaf:
                best_leaf_name, best_leaf_stats = best_eval_leaf
                xp.link.push_metrics({f"branch/eval_leaf/{epoch + 1}": {best_leaf_name: best_leaf_stats}})

            if dev_dl is not None:
                epoch_dev_metrics = {}
                for leaf, _ in sorted_leaves[: self.dev_eval_top_k]:
                    dev_metrics = self._evaluate_leaf_on_loader(
                        leaf, dev_dl, logger, tag=f"epoch {epoch + 1}"
                    )
                    if dev_metrics:
                        self.leaf_dev_metrics[leaf] = dev_metrics
                        epoch_dev_metrics[leaf] = dev_metrics
                if epoch_dev_metrics:
                    best_dev_leaf = self._best_leaf(epoch_dev_metrics)
                    if best_dev_leaf:
                        dev_leaf_name, dev_leaf_stats = best_dev_leaf
                        xp.link.push_metrics({f"branch/dev_leaf/{epoch + 1}": {dev_leaf_name: dev_leaf_stats}})
                        table = PrettyTable()
                        table.field_names = ["leaf", "precision", "recall", "f1"]
                        table.add_row(
                            [
                                dev_leaf_name,
                                f"{dev_leaf_stats.get('precision', 0.0):.4f}",
                                f"{dev_leaf_stats.get('recall', 0.0):.4f}",
                                f"{dev_leaf_stats.get('f1', 0.0):.4f}",
                            ]
                        )
                        logger.info("\nBest dev leaf (epoch %d):\n%s", epoch + 1, table.get_string())

            self._log_leaf_ranking(
                title=f"Leaf Factors Eval (epoch {epoch + 1})",
                metrics_dict=leaf_metrics,
                logger=logger,
                top_k=self.dev_eval_top_k,
                dev_metrics=self.leaf_dev_metrics,
                sorted_items=sorted_leaves,
            )

            total_eval_loss = selector_eval.get("total", 0.0) + expert_eval.get("total", 0.0)
            if total_eval_loss < best_eval:
                best_eval = total_eval_loss
                self._save_checkpoint(logger)

        if eval_dl is None:
            self._save_checkpoint(logger)

    def evaluate(self, loader, logger, log_to_xp: bool = True):
        disable_progress = should_disable_tqdm(metrics_only=True)
        selector_sums = defaultdict(float)
        selector_counts = defaultdict(int)
        expert_sums = defaultdict(float)
        expert_counts = defaultdict(int)
        label_names = loader.label_names if hasattr(loader, "label_names") else None
        label_groups = _build_label_groups(label_names)
        leaf_stats: dict[str, dict[str, int]] = {}
        leaf_per_class = {} if label_groups else None

        self._set_mode(train=False)

        with torch.no_grad():
            iterator = tqdm(loader, desc="Branch Composite Eval", disable=disable_progress)
            for batch in iterator:
                embeddings = batch["embeddings"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                incoming = batch.get("incoming")
                outgoing = batch.get("outgoing")
                ner_tags = batch.get("ner_tags")
                if incoming is not None:
                    incoming = incoming.to(self.device, non_blocking=True)
                if outgoing is not None:
                    outgoing = outgoing.to(self.device, non_blocking=True)
                if ner_tags is not None:
                    ner_tags = ner_tags.to(self.device, non_blocking=True)

                self._forward_batch(
                    embeddings,
                    attention_mask,
                    incoming,
                    outgoing,
                    selector_sums,
                    selector_counts,
                    expert_sums,
                    expert_counts,
                    training=False,
                    ner_tags=ner_tags,
                    leaf_stats=leaf_stats,
                    leaf_per_class=leaf_per_class,
                    label_groups=label_groups,
                )

        selector_avg = {k: selector_sums[k] / max(selector_counts[k], 1) for k in selector_sums}
        expert_avg = {k: expert_sums[k] / max(expert_counts[k], 1) for k in expert_sums}
        leaf_metrics = self._finalize_leaf_metrics(leaf_stats, leaf_per_class, label_groups)

        if log_to_xp:
            return {
                "selector_avg": selector_avg,
                "expert_avg": expert_avg,
                "leaf_metrics": leaf_metrics,
            }
        return {
            "selector_avg": selector_avg,
            "expert_avg": expert_avg,
            "leaf_metrics": leaf_metrics,
        }

    def _forward_batch(
        self,
        embeddings,
        mask,
        incoming,
        outgoing,
        selector_sums,
        selector_counts,
        expert_sums,
        expert_counts,
        *,
        training,
        ner_tags=None,
        leaf_stats=None,
        leaf_per_class=None,
        label_groups=None,
    ):
        selector_loss, expert_loss = self._forward_node(
            self.root,
            embeddings,
            mask,
            incoming,
            outgoing,
            selector_sums,
            selector_counts,
            expert_sums,
            expert_counts,
            training=training,
            ner_tags=ner_tags,
            leaf_stats=leaf_stats,
            leaf_per_class=leaf_per_class,
            label_groups=label_groups,
        )
        return selector_loss, expert_loss

    def _forward_node(
        self,
        node: BranchNode,
        embeddings,
        mask,
        incoming,
        outgoing,
        selector_sums,
        selector_counts,
        expert_sums,
        expert_counts,
        *,
        training,
        ner_tags=None,
        leaf_stats=None,
        leaf_per_class=None,
        label_groups=None,
    ):
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        selector_loss, selector_metrics, selector_out = self._selector_forward(
            node.selector, embeddings, mask, incoming, outgoing
        )
        for name, value in selector_metrics.items():
            selector_sums[name] += value
            selector_counts[name] += 1

        selection_mask = self._build_selection_mask(selector_out["gates"], mask)
        selected_embeddings = embeddings * selection_mask.unsqueeze(-1)
        selected_mask = (mask * selection_mask.long()).clamp(max=1)
        selected_incoming = incoming * selection_mask if incoming is not None else None
        selected_outgoing = outgoing * selection_mask if outgoing is not None else None

        expert_loss, expert_metrics, expert_out = self._expert_forward(
            node.expert, selected_embeddings, selected_mask, selected_incoming, selected_outgoing
        )
        for name, value in expert_metrics.items():
            expert_sums[name] += value
            expert_counts[name] += 1

        selector_total = selector_loss
        expert_total = expert_loss

        if node.children:
            predictions = expert_out["pi"].argmax(dim=-1)
            valid = selected_mask > 0
            for idx, child in enumerate(node.children):
                child_bool = valid & (predictions == idx)
                if not child_bool.any():
                    continue
                child_mask = child_bool.long()
                child_embeddings = selected_embeddings * child_bool.unsqueeze(-1).to(selected_embeddings.dtype)
                child_incoming = (
                    selected_incoming * child_bool if selected_incoming is not None else None
                )
                child_outgoing = (
                    selected_outgoing * child_bool if selected_outgoing is not None else None
                )
                c_sel, c_exp = self._forward_node(
                    child,
                    child_embeddings,
                    child_mask,
                    child_incoming,
                    child_outgoing,
                    selector_sums,
                    selector_counts,
                    expert_sums,
                    expert_counts,
                    training=training,
                    ner_tags=ner_tags,
                    leaf_stats=leaf_stats,
                    leaf_per_class=leaf_per_class,
                    label_groups=label_groups,
                )
                selector_total = selector_total + c_sel
                expert_total = expert_total + c_exp
        else:
            if ner_tags is not None and leaf_stats is not None:
                self._accumulate_leaf_stats(
                    leaf_stats,
                    leaf_per_class,
                    expert_out["pi"],
                    selected_mask,
                    ner_tags,
                    node.path,
                    label_groups,
                )

        return selector_total, expert_total

    def _selector_forward(self, selector, tokens, mask, incoming, outgoing):
        outputs = selector(tokens, mask, incoming, outgoing)
        h_anchor = outputs["h_anchor"]
        h_rat = outputs["h_rat"]
        h_comp = outputs["h_comp"]
        gates = outputs["gates"]
        null_vec = outputs.get("null")

        rat_loss = nt_xent(h_rat, h_anchor, temperature=self.selector_tau)
        comp_loss = complement_loss(
            h_comp,
            h_anchor,
            null_vec if self.selector_use_null else None,
            self.selector_use_null,
            temperature=self.selector_tau,
        )
        sparsity = rat_sparsity_loss(gates, mask)
        tv = total_variation_1d(gates, mask)

        total_loss = rat_loss
        total_loss = total_loss + self.selector_l_comp * comp_loss
        total_loss = total_loss + self.selector_l_s * sparsity
        total_loss = total_loss + self.selector_l_tv * tv

        metrics = {
            "rat": float(rat_loss.detach()),
            "comp": float(comp_loss.detach()),
            "sparsity": float(sparsity.detach()),
            "tv": float(tv.detach()),
            "total": float(total_loss.detach()),
        }
        return total_loss, metrics, outputs

    def _expert_forward(self, expert, embeddings, mask, incoming, outgoing):
        outputs = expert(embeddings, mask, incoming, outgoing)
        anchor = outputs["anchor"]
        reconstruction = outputs["reconstruction"]

        anchor = F.normalize(anchor, dim=-1)
        reconstruction = F.normalize(reconstruction, dim=-1)

        logits = anchor @ reconstruction.t() / max(self.contrastive_tau, 1e-6)
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_ab = F.cross_entropy(logits, targets)
        loss_ba = F.cross_entropy(logits.t(), targets)
        sent_loss = 0.5 * (loss_ab + loss_ba)

        token_reconstruction = outputs.get("token_reconstruction")
        if token_reconstruction is not None:
            mask_float = mask.unsqueeze(-1).to(dtype=token_reconstruction.dtype)
            diff = token_reconstruction - embeddings
            token_loss = (diff.pow(2) * mask_float).sum() / mask_float.sum().clamp_min(1.0)
        else:
            token_loss = embeddings.new_tensor(0.0)

        entropy_loss = outputs["entropy"].mean()
        overlap_loss = outputs["overlap"].mean()
        diversity_loss = outputs["diversity"]
        balance_loss = outputs["balance"]
        attention_loss = outputs.get("attention_entropy")
        if attention_loss is not None:
            attention_loss = attention_loss.mean()
        else:
            attention_loss = embeddings.new_tensor(0.0)

        loss_components = {
            "sent": sent_loss,
            "token": token_loss,
            "entropy": entropy_loss,
            "overlap": overlap_loss,
            "diversity": diversity_loss,
            "balance": balance_loss,
            "attention": attention_loss,
        }

        weights = self._expert_loss_weights()
        if "continuity" in weights:
            if "continuity" not in outputs:
                raise KeyError("Continuity weight configured but ExpertModel continuity output missing.")
            loss_components["continuity"] = outputs["continuity"].mean()

        total_loss = sum(weights[key] * loss_components[key] for key in loss_components)
        metrics = {key: float(value.detach()) for key, value in loss_components.items()}
        metrics["total"] = float(total_loss.detach())
        return total_loss, metrics, outputs

    def _expert_loss_weights(self):
        weights_cfg = self.model_cfg.loss_weights
        weights = {
            "sent": float(weights_cfg.sent),
            "token": float(weights_cfg.token),
            "entropy": float(weights_cfg.entropy),
            "overlap": float(weights_cfg.overlap),
            "diversity": float(weights_cfg.diversity),
            "balance": float(weights_cfg.balance),
            "attention": float(weights_cfg.attention),
        }
        if self.model_cfg.expert.use_continuity:
            weights["continuity"] = float(weights_cfg.continuity)
        return weights

    def _build_selection_mask(self, gates: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_float = attention_mask.to(dtype=gates.dtype)
        selection = (gates >= self.selector_threshold).to(dtype=gates.dtype) * mask_float
        selected_counts = selection.sum(dim=1)
        need_fallback = selected_counts == 0
        if need_fallback.any():
            masked_gates = gates.masked_fill(mask_float == 0, -1e9)
            top_indices = masked_gates.argmax(dim=1)
            rows = torch.arange(gates.size(0), device=gates.device)
            selection[rows[need_fallback], top_indices[need_fallback]] = 1.0
            selection = selection * mask_float
        return selection

    def _leaf_factor_name(self, path: tuple[int, ...], factor_idx: int) -> str:
        full_path = path + (factor_idx,)
        suffix = "_".join(str(idx) for idx in full_path) or "root"
        name = f"leaf_{suffix}"
        self.leaf_name_to_path.setdefault(name, full_path)
        return name

    def _accumulate_leaf_stats(
        self,
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

        for idx in range(self.num_factors):
            name = self._leaf_factor_name(path, idx)
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

    def _finalize_leaf_metrics(self, stats_dict, per_class_stats, label_groups):
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

    def _set_mode(self, train: bool):
        for node in self.nodes:
            if train:
                node.selector.train()
                node.expert.train()
            else:
                node.selector.eval()
                node.expert.eval()

    def _save_checkpoint(self, logger):
        state = {
            "selectors": {self._path_key(node.path): node.selector.state_dict() for node in self.nodes},
            "experts": {self._path_key(node.path): node.expert.state_dict() for node in self.nodes},
            "num_stages": self.num_stages,
        }
        torch.save(state, "branching_composite.pth", _use_new_zipfile_serialization=False)
        logger.info("Saved branching composite checkpoints to %s", os.getcwd())

    def _load_checkpoint(self, logger):
        path = "branching_composite.pth"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        state = torch.load(path, map_location=self.device)
        for node in self.nodes:
            key = self._path_key(node.path)
            node.selector.load_state_dict(state["selectors"][key])
            node.expert.load_state_dict(state["experts"][key])
        logger.info("Loaded branching composite checkpoints from %s", path)

    @staticmethod
    def _path_key(path: tuple[int, ...]) -> str:
        if not path:
            return "root"
        return "root_" + "_".join(str(idx) for idx in path)

    @staticmethod
    def _log_metrics_table(title: str, metrics_dict: dict[str, float], logger) -> None:
        if not metrics_dict:
            logger.info("%s: (no metrics)", title)
            return
        columns = sorted(metrics_dict.keys())
        table = PrettyTable()
        table.field_names = columns
        table.add_row([f"{metrics_dict[name]:.4f}" for name in columns])
        logger.info("\n%s:\n%s", title, table.get_string())

    @staticmethod
    def _log_f1_table(title: str, metrics_dict: dict[str, dict[str, float]], logger) -> None:
        if not metrics_dict:
            logger.info("%s: (no metrics)", title)
            return
        table = build_eval_table(metrics_dict)
        logger.info("\n%s:\n%s", title, table)

    def _log_leaf_ranking(
        self,
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
        items = sorted_items or self._sorted_leaves(metrics_dict)
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

    @staticmethod
    def _sorted_leaves(metrics_dict: dict[str, dict[str, float]]):
        return sorted(metrics_dict.items(), key=lambda item: item[1].get("f1", 0.0), reverse=True)

    @staticmethod
    def _best_leaf(metrics_dict: dict[str, dict[str, float]]):
        if not metrics_dict:
            return None
        return max(metrics_dict.items(), key=lambda item: item[1].get("f1", 0.0))

    def _evaluate_leaf_on_loader(self, leaf_name, loader, logger, tag: str | None = None):
        path = self.leaf_name_to_path.get(leaf_name)
        if path is None:
            logger.warning("Leaf %s not found; cannot run dev evaluation.", leaf_name)
            return None
        if len(path) != self.num_stages:
            logger.warning("Leaf %s path length mismatch; expected %d got %d.", leaf_name, self.num_stages, len(path))
            return None
        node_paths = [tuple(path[:i]) for i in range(self.num_stages)]
        nodes = []
        for node_path in node_paths:
            node = self.path_to_node.get(node_path)
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

        self._set_mode(train=False)
        iterator = tqdm(
            loader,
            desc=f"Leaf Eval {leaf_name}",
            disable=should_disable_tqdm(metrics_only=True),
        )
        with torch.no_grad():
            for batch in iterator:
                embeddings = batch["embeddings"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                incoming = batch.get("incoming")
                outgoing = batch.get("outgoing")
                ner_tags = batch.get("ner_tags")
                if incoming is not None:
                    incoming = incoming.to(self.device, non_blocking=True)
                if outgoing is not None:
                    outgoing = outgoing.to(self.device, non_blocking=True)
                if ner_tags is None:
                    continue
                ner_tags = ner_tags.to(self.device, non_blocking=True)

                cur_embeddings = embeddings
                cur_mask = attention_mask
                cur_incoming = incoming
                cur_outgoing = outgoing

                for stage_idx, node in enumerate(nodes):
                    _, _, selector_out = self._selector_forward(node.selector, cur_embeddings, cur_mask, cur_incoming, cur_outgoing)
                    selection_mask = self._build_selection_mask(selector_out["gates"], cur_mask)
                    cur_embeddings = cur_embeddings * selection_mask.unsqueeze(-1)
                    cur_mask = (cur_mask * selection_mask.long()).clamp(max=1)
                    cur_incoming = cur_incoming * selection_mask if cur_incoming is not None else None
                    cur_outgoing = cur_outgoing * selection_mask if cur_outgoing is not None else None

                    _, _, expert_out = self._expert_forward(
                        node.expert, cur_embeddings, cur_mask, cur_incoming, cur_outgoing
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
                        cur_incoming = (
                            cur_incoming * child_bool if cur_incoming is not None else None
                        )
                        cur_outgoing = (
                            cur_outgoing * child_bool if cur_outgoing is not None else None
                        )
                    else:
                        valid = cur_mask > 0
                        if not valid.any():
                            break
                        predictions = expert_out["pi"].argmax(dim=-1)
                        pred_idx = valid & (predictions == final_factor)
                        gold = (ner_tags > 0) & valid
                        stats["tp"] += (pred_idx & gold).sum().item()
                        stats["fp"] += (pred_idx & (~gold)).sum().item()
                        stats["fn"] += ((~pred_idx) & gold).sum().item()
                        if per_class is not None:
                            for label, indices in label_groups.items():
                                class_mask = torch.zeros_like(valid, dtype=torch.bool)
                                for label_id in indices:
                                    class_mask |= ner_tags == label_id
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


@hydra_main(config_path="conf", config_name="composite", version_base="1.1")
def main(cfg):
    logger = get_logger("train_branching_composite.log")
    xp = get_xp()
    logger.info(f"Exp signature: {xp.sig}")
    logger.info(repr(cfg))
    logger.info(f"Work dir: {os.getcwd()}")

    configure_runtime(cfg)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable, using CPU.")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cfg.device = device.type

    train_dl, eval_dl, dev_dl = initialize_dataloaders(cfg, logger)
    trainer = BranchingCompositeTrainer(cfg, device)

    if cfg.eval.eval_only:
        trainer._load_checkpoint(logger)
        results = trainer.evaluate(eval_dl, logger)
        trainer._log_metrics_table("Selector Eval", results["selector_avg"], logger)
        trainer._log_metrics_table("Expert Eval", results["expert_avg"], logger)
        leaf_metrics = results["leaf_metrics"]
        sorted_leaves = trainer._sorted_leaves(leaf_metrics)
        trainer._log_f1_table("Leaf Factors Eval", leaf_metrics, logger)
        if dev_dl is not None:
            epoch_dev = {}
            for leaf, _ in sorted_leaves[: trainer.dev_eval_top_k]:
                dev_metrics = trainer._evaluate_leaf_on_loader(leaf, dev_dl, logger, tag="eval_only")
                if dev_metrics:
                    trainer.leaf_dev_metrics[leaf] = dev_metrics
                    epoch_dev[leaf] = dev_metrics
            if epoch_dev:
                best_dev_leaf = trainer._best_leaf(epoch_dev)
                if best_dev_leaf:
                    dev_leaf_name, dev_leaf_stats = best_dev_leaf
                    xp.link.push_metrics({"branch/dev_leaf_eval_only": {dev_leaf_name: dev_leaf_stats}})
        trainer._log_leaf_ranking(
            "Leaf Factors Eval",
            leaf_metrics,
            logger,
            top_k=trainer.dev_eval_top_k,
            dev_metrics=trainer.leaf_dev_metrics,
            sorted_items=sorted_leaves,
        )
    else:
        trainer.train(train_dl, eval_dl, dev_dl, logger, xp)


if __name__ == "__main__":
    main()
