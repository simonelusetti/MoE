import os
from collections import defaultdict

import torch
import torch.nn.functional as F

from tqdm import tqdm

from dora import get_xp, hydra_main

from .data import initialize_dataloaders
from .models import ProductManifoldModel
from .utils import get_logger, should_disable_tqdm


def _mean_pool(embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=embeddings.dtype)
    masked = embeddings * mask
    denom = mask.sum(dim=1).clamp_min(1.0)
    return masked.sum(dim=1) / denom


class ProductTrainer:
    def __init__(self, cfg, token_dim: int, device: torch.device):
        self.cfg = cfg
        self.device = device
        model_cfg = cfg.product_model
        self.model = ProductManifoldModel(cfg, token_dim).to(self.device)

        optim_cfg = model_cfg.optim
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(optim_cfg.lr),
            weight_decay=float(optim_cfg.weight_decay),
            betas=tuple(optim_cfg.betas),
        )

        self.grad_clip = float(cfg.train.grad_clip)
        self.margin = float(model_cfg.margin)
        self.temperature = float(model_cfg.temperature)
        betas = model_cfg.betas
        self.beta_dis_spar = float(betas.dis_spar)
        self.beta_consistency = float(betas.consistency)
        self.beta_regularization = float(betas.regularization)

        warmup = model_cfg.warmup_epochs
        self.warmup_consistency = int(warmup.consistency)
        self.warmup_distance = int(warmup.distance)

        loss_weights = getattr(model_cfg, "losses", None)
        if loss_weights is None:
            self.lambda_geom_sparsity = 0.0
            self.lambda_geom_orthogonality = 0.0
        else:
            self.lambda_geom_sparsity = float(getattr(loss_weights, "geom_sparsity", 0.0))
            self.lambda_geom_orthogonality = float(getattr(loss_weights, "geom_orthogonality", 0.0))

    def train(self, train_dl, eval_dl, logger, xp):
        disable_progress = should_disable_tqdm()
        best_loss = float("inf")

        for epoch in range(self.cfg.train.epochs):
            train_metrics = defaultdict(float)
            self.model.train()
            iterator = tqdm(train_dl, desc=f"Product Train {epoch + 1}", disable=disable_progress)
            for batch in iterator:
                loss, metrics = self._step(batch, epoch)
                if not torch.isfinite(loss):
                    raise RuntimeError("Loss became non-finite during training.")

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                for name, value in metrics.items():
                    train_metrics[name] += value

            num_batches = max(len(train_dl), 1)
            train_metrics = {name: value / num_batches for name, value in train_metrics.items()}
            logger.info("Epoch %d train metrics: %s", epoch + 1, train_metrics)
            xp.link.push_metrics({f"product/train/{epoch + 1}": train_metrics})

            if eval_dl is None:
                continue
            eval_metrics, factor_metrics = self.evaluate(eval_dl)
            logger.info("Epoch %d eval metrics: %s", epoch + 1, eval_metrics)
            xp.link.push_metrics({f"product/eval/{epoch + 1}": eval_metrics})
            if factor_metrics:
                xp.link.push_metrics({f"product/factors/{epoch + 1}": factor_metrics})

            total_loss = eval_metrics.get("loss", float("inf"))
            if total_loss < best_loss:
                best_loss = total_loss
                self._save_checkpoint(logger)

        if eval_dl is None:
            # save final model if no evaluation loop ran
            self._save_checkpoint(logger)

    def evaluate(self, loader):
        disable_progress = should_disable_tqdm(metrics_only=True)
        metrics = defaultdict(float)
        factor_stats = [dict(tp=0, fp=0, fn=0) for _ in range(self.model.num_subspaces)]
        has_labels = False

        self.model.eval()
        with torch.no_grad():
            iterator = tqdm(loader, desc="Product Eval", disable=disable_progress)
            for batch in iterator:
                _, step_metrics, out_a, anchors_mask = self._loss_components(
                    batch, epoch=None, training=False, return_anchor_outputs=True
                )
                for name, value in step_metrics.items():
                    metrics[name] += value

                if "ner_tags" in batch:
                    has_labels = True
                    ner_tags = batch["ner_tags"].to(self.device)
                    self._accumulate_factor_stats(
                        factor_stats, out_a["subspaces"], anchors_mask, ner_tags
                    )

        num_batches = max(len(loader), 1)
        avg_metrics = {name: value / num_batches for name, value in metrics.items()}
        factor_metrics = self._finalize_factor_metrics(factor_stats) if has_labels else {}
        return avg_metrics, factor_metrics

    def _step(self, batch, epoch: int):
        loss, metrics = self._loss_components(batch, epoch, training=True)
        return loss, metrics

    def _loss_components(
        self, batch, epoch: int | None, training: bool, *, return_anchor_outputs: bool = False
    ):
        tokens = batch["embeddings"].to(self.device)
        mask = batch["attention_mask"].to(self.device)

        if tokens.size(0) < 2:
            raise ValueError("Batch size must be >= 2 to form pairs.")

        anchors_tokens = tokens
        anchors_mask = mask
        paired_tokens = tokens.roll(shifts=1, dims=0)
        paired_mask = mask.roll(shifts=1, dims=0)

        anchor_sent = _mean_pool(anchors_tokens, anchors_mask)
        paired_sent = _mean_pool(paired_tokens, paired_mask)

        out_a = self.model(anchors_tokens, anchors_mask)
        out_b = self.model(paired_tokens, paired_mask)

        recon_loss = self._reconstruction_loss(anchor_sent, out_a["sentence_reconstruction"])
        recon_loss = recon_loss + self._reconstruction_loss(paired_sent, out_b["sentence_reconstruction"])
        recon_loss = recon_loss * 0.5

        token_loss = 0.5 * (
            self._token_reconstruction_loss(anchors_tokens, out_a["token_reconstruction"], anchors_mask)
            + self._token_reconstruction_loss(paired_tokens, out_b["token_reconstruction"], paired_mask)
        )

        sparsity_a = self._sparsity_loss(out_a["subspaces"], anchors_mask)
        sparsity_b = self._sparsity_loss(out_b["subspaces"], paired_mask)
        sparsity_loss = 0.5 * (sparsity_a + sparsity_b)

        pair_mask = anchors_mask & paired_mask
        distance_loss, delta = self._distance_loss(out_a["subspaces"], out_b["subspaces"], pair_mask)
        consistency_loss = self._consistency_loss(out_a, out_b)
        regularization_loss = self._regularization_loss(delta)

        geom_sparsity = anchor_sent.new_tensor(0.0)
        geom_orthogonality = anchor_sent.new_tensor(0.0)
        if self.lambda_geom_sparsity > 0 or self.lambda_geom_orthogonality > 0:
            sub_a = out_a["subspaces"]
            sub_b = out_b["subspaces"]
            if self.lambda_geom_sparsity > 0:
                geom_sparsity = 0.5 * (
                    self._geometric_sparsity(sub_a, anchors_mask)
                    + self._geometric_sparsity(sub_b, paired_mask)
                )
            if self.lambda_geom_orthogonality > 0:
                geom_orthogonality = 0.5 * (
                    self._geometric_orthogonality(sub_a, anchors_mask)
                    + self._geometric_orthogonality(sub_b, paired_mask)
                )

        dis_scale = self._warmup_scale(epoch, self.warmup_distance)
        cons_scale = self._warmup_scale(epoch, self.warmup_consistency)

        total_loss = recon_loss
        total_loss = total_loss + token_loss
        total_loss = total_loss + self.beta_dis_spar * dis_scale * (distance_loss + sparsity_loss)
        total_loss = total_loss + self.beta_consistency * cons_scale * consistency_loss
        total_loss = total_loss + self.beta_regularization * regularization_loss
        total_loss = total_loss + self.lambda_geom_sparsity * geom_sparsity
        total_loss = total_loss + self.lambda_geom_orthogonality * geom_orthogonality

        metrics = {
            "loss": float(total_loss.detach()),
            "recon": float(recon_loss.detach()),
            "token_recon": float(token_loss.detach()),
            "distance": float(distance_loss.detach()),
            "sparsity": float(sparsity_loss.detach()),
            "consistency": float(consistency_loss.detach()),
            "regularization": float(regularization_loss.detach()),
            "geom_sparsity": float(geom_sparsity.detach()),
            "geom_orthogonality": float(geom_orthogonality.detach()),
        }
        if return_anchor_outputs:
            return total_loss, metrics, out_a, anchors_mask
        return total_loss, metrics

    @staticmethod
    def _warmup_scale(epoch: int | None, warmup_epochs: int) -> float:
        if epoch is None or warmup_epochs <= 0:
            return 1.0
        return min(1.0, (epoch + 1) / warmup_epochs)

    def _reconstruction_loss(self, target: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        target_norm = F.normalize(target, dim=-1)
        recon_norm = F.normalize(reconstruction, dim=-1)
        cosine = (target_norm * recon_norm).sum(dim=-1)
        return (1.0 - cosine).mean()

    def _token_reconstruction_loss(
        self, original: torch.Tensor, reconstructed: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        diff = (reconstructed - original).pow(2).sum(dim=-1)
        mask_float = mask.to(diff.dtype)
        return (diff * mask_float).sum() / mask_float.sum().clamp_min(1.0)

    def _sparsity_loss(self, subspaces: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_float = mask.to(subspaces.dtype)
        masked = subspaces * mask_float.unsqueeze(-1).unsqueeze(-1)
        others = masked.sum(dim=2, keepdim=True) - masked
        penalty = (masked * others).abs().sum(dim=-1)
        denom = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        penalty = penalty.sum(dim=(1, 2)) / (denom * subspaces.size(2))
        return penalty.mean()

    def _distance_loss(
        self, sub_a: torch.Tensor, sub_b: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_float = mask.to(sub_a.dtype)
        diff = torch.norm(sub_a - sub_b, dim=-1)
        denom = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        diff_mean = (diff * mask_float.unsqueeze(-1)).sum(dim=1) / denom

        norm_a = torch.norm(sub_a, dim=-1)
        norm_b = torch.norm(sub_b, dim=-1)
        avg_norm = (
            (norm_a * mask_float.unsqueeze(-1)).sum(dim=1)
            + (norm_b * mask_float.unsqueeze(-1)).sum(dim=1)
        ) / (2.0 * denom)

        delta = diff_mean / avg_norm.clamp_min(1e-6)

        alpha_idx = delta.argmax(dim=1)
        alpha = F.one_hot(alpha_idx, num_classes=sub_a.size(2)).to(delta.dtype)

        margin_term = F.relu(self.margin - delta)
        contrastive = alpha * margin_term.pow(2)
        similarity = (1.0 - alpha) * delta.pow(2)
        loss = (contrastive + similarity).sum(dim=1).mean()
        return loss, delta

    def _consistency_loss(self, out_a: dict, out_b: dict) -> torch.Tensor:
        sub_a = out_a["subspaces"]
        sub_b = out_b["subspaces"]
        mask_a = out_a["mask"].to(sub_a.dtype)
        mask_b = out_b["mask"].to(sub_b.dtype)

        total_a = sub_a.sum(dim=2)
        total_b = sub_b.sum(dim=2)

        losses = []
        for idx in range(sub_a.size(2)):
            mixed_ab = sub_a[:, :, idx, :] + (total_b - sub_b[:, :, idx, :])
            mixed_ba = sub_b[:, :, idx, :] + (total_a - sub_a[:, :, idx, :])

            dec_ab = self.model.decode_tokens(mixed_ab)
            dec_ba = self.model.decode_tokens(mixed_ba)

            rec_ab = self.model.encode_tokens(dec_ab, out_a["mask"])["subspaces"][:, :, idx, :]
            rec_ba = self.model.encode_tokens(dec_ba, out_b["mask"])["subspaces"][:, :, idx, :]

            diff_ab = (rec_ab - sub_a[:, :, idx, :]).pow(2).sum(dim=-1)
            diff_ba = (rec_ba - sub_b[:, :, idx, :]).pow(2).sum(dim=-1)

            loss_ab = ((diff_ab * mask_a).sum(dim=1) / mask_a.sum(dim=1).clamp_min(1.0)).mean()
            loss_ba = ((diff_ba * mask_b).sum(dim=1) / mask_b.sum(dim=1).clamp_min(1.0)).mean()

            losses.append(loss_ab)
            losses.append(loss_ba)

        if not losses:
            return torch.zeros((), device=self.device)
        return torch.stack(losses).mean()

    def _regularization_loss(self, delta: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(delta / self.temperature, dim=1)
        expectation = weights.mean(dim=0)
        uniform = torch.full_like(expectation, 1.0 / expectation.numel())
        return (expectation - uniform).pow(2).sum()

    @staticmethod
    def _geometric_sparsity(subspaces: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_float = mask.to(subspaces.dtype)
        masked = subspaces * mask_float.unsqueeze(-1).unsqueeze(-1)
        others = masked.sum(dim=2, keepdim=True) - masked
        penalty = (masked * others).abs().sum(dim=-1)
        denom = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        penalty = penalty.sum(dim=(1, 2)) / (denom * subspaces.size(2))
        return penalty.mean()

    @staticmethod
    def _geometric_orthogonality(subspaces: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_float = mask.to(subspaces.dtype)
        masked = subspaces * mask_float.unsqueeze(-1).unsqueeze(-1)
        token_counts = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        factor_means = masked.sum(dim=1) / token_counts.unsqueeze(-1)
        centered = factor_means - factor_means.mean(dim=1, keepdim=True)
        gram = torch.einsum("bkd,bjd->bkj", centered, centered)
        diag = torch.diagonal(gram, dim1=1, dim2=2)
        off_diag = gram - torch.diag_embed(diag)
        denom = max(subspaces.size(2) * (subspaces.size(2) - 1), 1)
        return (off_diag.pow(2).sum(dim=(1, 2)) / denom).mean()

    def _accumulate_factor_stats(
        self,
        stats: list[dict[str, int]],
        subspaces: torch.Tensor,
        mask: torch.Tensor,
        ner_tags: torch.Tensor,
    ) -> None:
        strengths = torch.norm(subspaces, dim=-1)  # [B, T, K]
        predictions = strengths.argmax(dim=-1)  # [B, T]

        valid = mask > 0
        gold = (ner_tags > 0) & valid

        for idx in range(self.model.num_subspaces):
            pred_idx = (predictions == idx) & valid
            tp = (pred_idx & gold).sum().item()
            fp = (pred_idx & (~gold)).sum().item()
            fn = ((~pred_idx) & gold).sum().item()

            stats[idx]["tp"] += tp
            stats[idx]["fp"] += fp
            stats[idx]["fn"] += fn

    def _finalize_factor_metrics(self, stats: list[dict[str, int]]) -> dict[str, dict[str, float]]:
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

    def _save_checkpoint(self, logger):
        path = "product_model.pth"
        torch.save(self.model.state_dict(), path, _use_new_zipfile_serialization=False)
        logger.info("Saved product-manifold model to %s", os.path.abspath(path))


@hydra_main(config_path="conf", config_name="product", version_base="1.1")
def main(cfg):
    logger = get_logger("train_product.log")
    xp = get_xp()
    logger.info("Exp signature: %s", xp.sig)
    logger.info("Config: %s", cfg)
    logger.info("Working directory: %s", os.getcwd())

    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_dl, eval_dl = initialize_dataloaders(cfg, logger)
    train_dl = train_dl  # explicit for clarity

    input_dim = int(cfg.product_model.input_dim)
    if input_dim <= 0:
        sample_batch = next(iter(train_dl))
        input_dim = sample_batch["embeddings"].size(-1)

    trainer = ProductTrainer(cfg, input_dim, device)

    if cfg.eval.eval_only:
        eval_metrics, factor_metrics = trainer.evaluate(eval_dl)
        logger.info("Eval metrics: %s", eval_metrics)
        xp.link.push_metrics({"product/eval": eval_metrics})
        if factor_metrics:
            xp.link.push_metrics({"product/factors": factor_metrics})
    else:
        trainer.train(train_dl, eval_dl, logger, xp)


if __name__ == "__main__":
    main()
