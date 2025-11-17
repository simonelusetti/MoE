import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from prettytable import PrettyTable
from tqdm import tqdm

from dora import get_xp, hydra_main

# Make the RatCon package (sibling repo) importable.
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


class CompositeTrainer:
    """
    Joint trainer that first selects rationales with the RatCon selector model,
    then feeds the selected tokens to the ExpertModel mixture-of-experts stage.
    """

    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.selector_cfg = cfg.selector_model
        self.selector = RationaleSelectorModel(self.selector_cfg).to(self.device)
        self.expert = ExpertModel(cfg.model).to(self.device)

        selector_optim_cfg = self.selector_cfg.optim
        self.selector_optimizer = torch.optim.AdamW(
            self.selector.parameters(),
            lr=float(selector_optim_cfg.lr),
            weight_decay=float(selector_optim_cfg.weight_decay),
            betas=tuple(selector_optim_cfg.betas),
        )

        expert_optim_cfg = cfg.model.optim
        self.expert_optimizer = torch.optim.AdamW(
            self.expert.parameters(),
            lr=expert_optim_cfg.lr,
            weight_decay=expert_optim_cfg.weight_decay,
            betas=expert_optim_cfg.betas,
        )

        self.grad_clip = float(cfg.train.grad_clip)
        self.selector_weight = float(cfg.composite.loss_weights.selector)
        self.expert_weight = float(cfg.composite.loss_weights.expert)
        self.contrastive_tau = float(cfg.model.contrastive_tau)

        loss_cfg = self.selector_cfg.loss
        self.selector_tau = float(loss_cfg.tau)
        self.selector_l_comp = float(loss_cfg.l_comp)
        self.selector_l_s = float(loss_cfg.l_s)
        self.selector_l_tv = float(loss_cfg.l_tv)
        self.selector_use_null = bool(loss_cfg.use_null_target)

        self.selector_threshold = float(cfg.composite.selector_threshold)

    def train(self, train_dl, eval_dl, logger, xp):
        disable_progress = should_disable_tqdm()
        best_eval = float("inf")

        for epoch in range(self.cfg.train.epochs):
            selector_sums = defaultdict(float)
            expert_sums = defaultdict(float)
            selector_counts = defaultdict(int)
            expert_counts = defaultdict(int)

            self.selector.train()
            self.expert.train()

            iterator = tqdm(train_dl, desc=f"Composite Train {epoch + 1}", disable=disable_progress)
            for batch in iterator:
                (
                    selector_loss,
                    selector_metrics,
                    selector_out,
                    embeddings,
                    attention_mask,
                ) = self._selector_forward(batch, training=True, return_outputs=True)

                selection_mask = self._build_selection_mask(selector_out["gates"], attention_mask)
                selected_embeddings = embeddings * selection_mask.unsqueeze(-1)
                selected_attention_mask = (attention_mask * selection_mask.long()).clamp(max=1)

                incoming = batch.get("incoming")
                outgoing = batch.get("outgoing")
                if incoming is not None:
                    incoming = incoming.to(self.device, non_blocking=True)
                if outgoing is not None:
                    outgoing = outgoing.to(self.device, non_blocking=True)

                expert_loss, expert_metrics, _ = self._expert_forward(
                    selected_embeddings,
                    selected_attention_mask,
                    incoming,
                    outgoing,
                )

                total_loss = self.selector_weight * selector_loss + self.expert_weight * expert_loss

                self.selector_optimizer.zero_grad(set_to_none=True)
                self.expert_optimizer.zero_grad(set_to_none=True)
                total_loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.selector.parameters(), self.grad_clip)
                    torch.nn.utils.clip_grad_norm_(self.expert.parameters(), self.grad_clip)

                self.selector_optimizer.step()
                self.expert_optimizer.step()

                selector_metrics_with_loss = dict(selector_metrics)
                selector_metrics_with_loss["loss"] = float(selector_loss.detach())
                expert_metrics_with_loss = dict(expert_metrics)
                expert_metrics_with_loss["loss"] = float(expert_loss.detach())

                for name, value in selector_metrics_with_loss.items():
                    selector_sums[name] += value
                    selector_counts[name] += 1
                for name, value in expert_metrics_with_loss.items():
                    expert_sums[name] += value
                    expert_counts[name] += 1

            selector_avg = {name: selector_sums[name] / max(selector_counts[name], 1) for name in selector_sums}
            expert_avg = {name: expert_sums[name] / max(expert_counts[name], 1) for name in expert_sums}

            xp.link.push_metrics({f"composite/train_selector/{epoch + 1}": selector_avg})
            xp.link.push_metrics({f"composite/train_expert/{epoch + 1}": expert_avg})

            self._log_metrics_table(f"Selector Train (epoch {epoch + 1})", selector_avg, logger)

            if eval_dl is None:
                self._log_metrics_table(f"Expert Train (epoch {epoch + 1})", expert_avg, logger)
                continue

            eval_results = self.evaluate(eval_dl, logger)
            selector_eval_avg = eval_results["selector_avg"]
            expert_eval_avg = eval_results["expert_avg"]

            self._log_f1_table("Selector Eval", eval_results["selector_factor_metrics"], logger)
            self._log_metrics_table(f"Expert Train (epoch {epoch + 1})", expert_avg, logger)
            self._log_f1_table("Expert Eval", eval_results["expert_factor_metrics"], logger)

            xp.link.push_metrics({f"composite/eval/{epoch + 1}": eval_results["metrics"]})

            total_eval_loss = selector_eval_avg.get("loss", 0.0) + expert_eval_avg.get("loss", 0.0)
            if total_eval_loss < best_eval:
                best_eval = total_eval_loss
                self._save_checkpoints(logger)

        if eval_dl is None:
            self._save_checkpoints(logger)

    def evaluate(self, loader, logger):
        disable_progress = should_disable_tqdm(metrics_only=True)

        selector_sums = defaultdict(float)
        expert_sums = defaultdict(float)
        selector_counts = defaultdict(int)
        expert_counts = defaultdict(int)

        selector_stats = dict(tp=0, fp=0, fn=0)
        expert_stats = [dict(tp=0, fp=0, fn=0) for _ in range(self.expert.num_experts)]
        label_names = loader.label_names if hasattr(loader, "label_names") else None
        label_groups = _build_label_groups(label_names)
        expert_class_stats = (
            [
                {label: dict(tp=0, fp=0, fn=0) for label in label_groups.keys()}
                for _ in range(self.expert.num_experts)
            ]
            if label_groups
            else None
        )

        self.selector.eval()
        self.expert.eval()

        has_labels = False

        with torch.no_grad():
            iterator = tqdm(loader, desc="Composite Eval", disable=disable_progress)
            for batch in iterator:
                (
                    selector_loss,
                    selector_metrics,
                    selector_out,
                    embeddings,
                    attention_mask,
                ) = self._selector_forward(batch, training=False, return_outputs=True)

                selection_mask = self._build_selection_mask(selector_out["gates"], attention_mask)
                selected_embeddings = embeddings * selection_mask.unsqueeze(-1)
                selected_attention_mask = (attention_mask * selection_mask.long()).clamp(max=1)

                incoming = batch.get("incoming")
                outgoing = batch.get("outgoing")
                if incoming is not None:
                    incoming = incoming.to(self.device, non_blocking=True)
                if outgoing is not None:
                    outgoing = outgoing.to(self.device, non_blocking=True)

                expert_loss, expert_metrics, expert_out = self._expert_forward(
                    selected_embeddings,
                    selected_attention_mask,
                    incoming,
                    outgoing,
                )

                selector_metrics_with_loss = dict(selector_metrics)
                selector_metrics_with_loss["loss"] = float(selector_loss.detach())
                expert_metrics_with_loss = dict(expert_metrics)
                expert_metrics_with_loss["loss"] = float(expert_loss.detach())

                for name, value in selector_metrics_with_loss.items():
                    selector_sums[name] += value
                    selector_counts[name] += 1
                for name, value in expert_metrics_with_loss.items():
                    expert_sums[name] += value
                    expert_counts[name] += 1

                if "ner_tags" in batch:
                    has_labels = True
                    ner_tags = batch["ner_tags"].to(self.device, non_blocking=True)
                    self._accumulate_selector_stats(selector_stats, selector_out["gates"], attention_mask, ner_tags)
                    self._accumulate_expert_stats(
                        expert_stats,
                        expert_out["pi"],
                        selected_attention_mask,
                        ner_tags,
                        label_groups=label_groups,
                        per_class_stats=expert_class_stats,
                    )

        selector_avg = {name: selector_sums[name] / max(selector_counts[name], 1) for name in selector_sums}
        expert_avg = {name: expert_sums[name] / max(expert_counts[name], 1) for name in expert_sums}

        if has_labels:
            selector_metrics = self._finalize_selector_metrics(selector_stats)
            expert_metrics = self._finalize_expert_metrics(
                expert_stats, per_class_stats=expert_class_stats, label_groups=label_groups
            )
        else:
            selector_metrics = {}
            expert_metrics = {}

        eval_metrics = {
            **{f"selector/{k}": v for k, v in selector_avg.items()},
            **{f"expert/{k}": v for k, v in expert_avg.items()},
        }

        for factor, stats in selector_metrics.items():
            eval_metrics[f"selector/precision/{factor}"] = stats["precision"]
            eval_metrics[f"selector/recall/{factor}"] = stats["recall"]
            eval_metrics[f"selector/f1/{factor}"] = stats["f1"]
        for factor, stats in expert_metrics.items():
            eval_metrics[f"expert/precision/{factor}"] = stats["precision"]
            eval_metrics[f"expert/recall/{factor}"] = stats["recall"]
            eval_metrics[f"expert/f1/{factor}"] = stats["f1"]

        return {
            "selector_avg": selector_avg,
            "expert_avg": expert_avg,
            "selector_factor_metrics": selector_metrics,
            "expert_factor_metrics": expert_metrics,
            "metrics": eval_metrics,
        }

    def _selector_forward(self, batch, training: bool, return_outputs: bool = False):
        tokens = batch["embeddings"].to(self.device, non_blocking=True)
        mask = batch["attention_mask"].to(self.device, non_blocking=True)

        incoming = batch.get("incoming")
        outgoing = batch.get("outgoing")
        if incoming is not None:
            incoming = incoming.to(self.device, non_blocking=True)
        if outgoing is not None:
            outgoing = outgoing.to(self.device, non_blocking=True)

        outputs = self.selector(tokens, mask, incoming, outgoing)
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

        if return_outputs:
            return total_loss, metrics, outputs, tokens, mask
        return total_loss, metrics

    def _expert_forward(self, embeddings, attention_mask, incoming, outgoing):
        outputs = self.expert(embeddings, attention_mask, incoming, outgoing)
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
            mask = attention_mask.unsqueeze(-1).to(dtype=token_reconstruction.dtype)
            diff = token_reconstruction - embeddings
            token_loss = (diff.pow(2) * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            token_loss = embeddings.new_tensor(0.0)

        entropy_loss = outputs["entropy"].mean()
        overlap_loss = outputs["overlap"].mean()
        diversity_loss = outputs["diversity"]
        balance_loss = outputs["balance"]

        loss_components = {
            "sent": sent_loss,
            "token": token_loss,
            "entropy": entropy_loss,
            "overlap": overlap_loss,
            "diversity": diversity_loss,
            "balance": balance_loss,
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
        weights_cfg = self.cfg.model.loss_weights
        weights = {
            "sent": float(weights_cfg.sent),
            "token": float(weights_cfg.token),
            "entropy": float(weights_cfg.entropy),
            "overlap": float(weights_cfg.overlap),
            "diversity": float(weights_cfg.diversity),
            "balance": float(weights_cfg.balance),
        }
        if self.cfg.model.expert.use_continuity:
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

    def _accumulate_selector_stats(
        self, stats: dict[str, int], gates: torch.Tensor, mask: torch.Tensor, ner_tags: torch.Tensor
    ) -> None:
        valid = mask > 0
        gold = (ner_tags > 0) & valid

        predictions = (gates >= self.selector_threshold) & valid
        tp = (predictions & gold).sum().item()
        fp = (predictions & (~gold)).sum().item()
        fn = ((~predictions) & gold).sum().item()

        stats["tp"] += tp
        stats["fp"] += fp
        stats["fn"] += fn

    @staticmethod
    def _accumulate_expert_stats(
        stats: list[dict[str, int]],
        routing: torch.Tensor,
        mask: torch.Tensor,
        ner_tags: torch.Tensor,
        *,
        label_groups=None,
        per_class_stats=None,
    ) -> None:
        predictions = routing.argmax(dim=-1)

        valid = mask > 0
        gold = (ner_tags > 0) & valid

        gold_by_class = {}
        if label_groups:
            for label, indices in label_groups.items():
                class_mask = torch.zeros_like(valid, dtype=torch.bool)
                for label_id in indices:
                    class_mask |= ner_tags == label_id
                gold_by_class[label] = class_mask & valid

        for idx in range(routing.size(-1)):
            pred_idx = (predictions == idx) & valid
            tp = (pred_idx & gold).sum().item()
            fp = (pred_idx & (~gold)).sum().item()
            fn = ((~pred_idx) & gold).sum().item()

            stats[idx]["tp"] += tp
            stats[idx]["fp"] += fp
            stats[idx]["fn"] += fn

            if per_class_stats and gold_by_class:
                for label, class_mask in gold_by_class.items():
                    tp_c = (pred_idx & class_mask).sum().item()
                    fp_c = (pred_idx & (~class_mask)).sum().item()
                    fn_c = ((~pred_idx) & class_mask).sum().item()

                    cls_counts = per_class_stats[idx][label]
                    cls_counts["tp"] += tp_c
                    cls_counts["fp"] += fp_c
                    cls_counts["fn"] += fn_c

    @staticmethod
    def _finalize_selector_metrics(stats: dict[str, int]) -> dict[str, dict[str, float]]:
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {
            "rationale": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        }

    @staticmethod
    def _finalize_expert_metrics(
        stats: list[dict[str, int]], *, per_class_stats=None, label_groups=None
    ) -> dict[str, dict[str, float]]:
        results = {}
        for idx, counts in enumerate(stats):
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            result = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
            if per_class_stats and label_groups:
                per_class_metrics = {}
                for label in label_groups.keys():
                    cls_counts = per_class_stats[idx][label]
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
            results[f"expert_{idx}"] = result
        return results

    def _save_checkpoints(self, logger):
        torch.save(self.selector.state_dict(), "composite_selector.pth", _use_new_zipfile_serialization=False)
        torch.save(self.expert.state_dict(), "composite_expert.pth", _use_new_zipfile_serialization=False)
        logger.info("Saved composite checkpoints to %s", os.getcwd())

    @staticmethod
    def _log_f1_table(title: str, metrics_dict: dict[str, dict[str, float]], logger) -> None:
        if not metrics_dict:
            logger.info("%s F1: (no labels available)", title)
            return
        table = build_eval_table(metrics_dict)
        logger.info("\n%s precision/recall/F1:\n%s", title, table)

    @staticmethod
    def _log_metrics_table(title: str, metrics_dict: dict[str, float], logger) -> None:
        if not metrics_dict:
            logger.info("%s: (no metrics)", title)
            return
        table = PrettyTable()
        columns = sorted(metrics_dict)
        table.field_names = columns
        table.add_row([f"{metrics_dict[col]:.4f}" for col in columns])
        logger.info("\n%s:\n%s", title, table.get_string())


@hydra_main(config_path="conf", config_name="composite", version_base="1.1")
def main(cfg):
    logger = get_logger("train_composite.log")
    xp = get_xp()
    logger.info("Exp signature: %s", xp.sig)
    logger.info("Config: %s", cfg)
    logger.info("Working directory: %s", os.getcwd())

    configure_runtime(cfg)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cfg.device = device.type

    train_dl, eval_dl, _ = initialize_dataloaders(cfg, logger)

    trainer = CompositeTrainer(cfg, device)
    trainer.train(train_dl, eval_dl, logger, xp)


if __name__ == "__main__":
    main()
