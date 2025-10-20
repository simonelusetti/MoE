import os
from collections import defaultdict

import torch
import torch.nn.functional as F

from tqdm import tqdm

from dora import get_xp, hydra_main

from .data import initialize_dataloaders
from .models import ExpertModel
from .utils import get_logger, should_disable_tqdm
from .metrics import build_train_table, build_eval_table, evaluate_factor_metrics


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

class ExpertTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.weights = self._prepare_expert_weights(cfg)
        self.contrastive_tau = getattr(cfg.model, "contrastive_tau", 0.07)

    @staticmethod
    def _prepare_expert_weights(cfg):
        weights_cfg = cfg.model.loss_weights
        return {
            "sent": float(weights_cfg.sent),
            "token": float(weights_cfg.token),
            "entropy": float(weights_cfg.entropy),
            "overlap": float(weights_cfg.overlap),
            "diversity": float(weights_cfg.diversity),
            "balance": float(weights_cfg.balance),
        }

    def _loss(self, model, batch, device):
        embeddings = batch["embeddings"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        incoming = batch.get("incoming")
        outgoing = batch.get("outgoing")
        if incoming is not None:
            incoming = incoming.to(device, non_blocking=True)
        if outgoing is not None:
            outgoing = outgoing.to(device, non_blocking=True)

        outputs = model(embeddings, attention_mask, incoming, outgoing)
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

        total_loss = sum(self.weights[key] * loss_components[key] for key in loss_components)
        metrics = {key: float(value.detach()) for key, value in loss_components.items()}
        metrics["total"] = float(total_loss.detach())
        return total_loss, metrics

    def _run_epoch(
        self,
        model,
        loader,
        device,
        *,
        train=False,
        optimizer=None,
        grad_clip=0.0,
        desc="",
        disable_progress=False,
    ):
        if train and optimizer is None:
            raise ValueError("Optimizer must be provided when train=True.")

        sums = defaultdict(float)
        counts = defaultdict(int)

        model.train() if train else model.eval()
        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            iterator = tqdm(loader, desc=desc, disable=disable_progress)
            for batch in iterator:
                if train:
                    optimizer.zero_grad(set_to_none=True)
                total_loss, metrics = self._loss(model, batch, device)
                if train:
                    total_loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                for name, value in metrics.items():
                    sums[name] += value
                    counts[name] += 1

        return {name: (sums[name] / counts[name]) if counts[name] else 0.0 for name in sums}


    def _save_checkpoint(self, model, path, logger):
        torch.save(model.state_dict(), path, _use_new_zipfile_serialization=False)
        logger.info(f"Saved ExpertModel checkpoint to {path}")


    def _load_checkpoint(self, model, path, device, logger):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert checkpoint not found at {path}")
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        logger.info(f"Loaded ExpertModel checkpoint from {path}")


    def train(self, cfg, logger, train_dl, eval_dl, xp):
        device = cfg.device
        model = ExpertModel(cfg.model).to(device)

        optim_cfg = cfg.model.optim
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optim_cfg.lr,
            weight_decay=optim_cfg.weight_decay,
            betas=optim_cfg.betas,
        )
        grad_clip = cfg.train.grad_clip
        disable_progress = should_disable_tqdm()

        checkpoint_path = "expert_model.pth"
        best_f1 = float("-inf")
        best_epoch = None
        best_factor = None

        for epoch in range(cfg.train.epochs):
            train_metrics = self._run_epoch(
                model,
                train_dl,
                device,
                train=True,
                optimizer=optimizer,
                grad_clip=grad_clip,
                desc=f"Expert Train {epoch + 1}",
                disable_progress=disable_progress,
            )

            train_table = build_train_table(train_metrics, model=model, weights=self.weights)
            logger.info("\nTrain metrics (epoch %d):\n%s", epoch + 1, train_table)

            xp.link.push_metrics({f"expert/train/{epoch + 1}": train_metrics})

            eval_metrics = self._run_epoch(
                model,
                eval_dl,
                device,
                train=False,
                desc=f"Expert Eval {epoch + 1}",
                disable_progress=False,
            )
            factor_metrics = evaluate_factor_metrics(model, eval_dl, device, logger)

            eval_table = build_eval_table(factor_metrics)
            logger.info("\nEval factor metrics (epoch %d):\n%s", epoch + 1, eval_table)

            xp.link.push_metrics({f"expert/eval/{epoch + 1}": eval_metrics})
            if factor_metrics:
                xp.link.push_metrics({f"expert/factors/{epoch + 1}": factor_metrics})

            top_factor = None
            top_f1 = float("-inf")
            for factor, stats in (factor_metrics or {}).items():
                f1 = stats.get("f1", 0.0)
                if f1 > top_f1:
                    top_f1 = f1
                    top_factor = factor

            if top_factor is not None and top_f1 > best_f1:
                best_f1 = top_f1
                best_epoch = epoch + 1
                best_factor = top_factor
                self._save_checkpoint(model, checkpoint_path, logger)

        if best_epoch is None:
            self._save_checkpoint(model, checkpoint_path, logger)
            logger.info("No factor improvements detected; saved final model.")
        else:
            logger.info(
                "Best epoch %d with factor %s achieving F1 %.4f",
                best_epoch,
                best_factor,
                best_f1,
            )
            xp.link.push_metrics({"expert/best_epoch": best_epoch, "expert/best_factor": best_factor, "expert/best_f1": best_f1})

        return model


    def evaluate(self, cfg, logger, eval_dl, xp):
        device = cfg.device
        model = ExpertModel(cfg.model).to(device)
        checkpoint_path = "expert_model.pth"
        self._load_checkpoint(model, checkpoint_path, device, logger)

        eval_metrics = self._run_epoch(
            model,
            eval_dl,
            device,
            train=False,
            desc="Expert Evaluation",
            disable_progress=False,
        )
        factor_metrics = evaluate_factor_metrics(model, eval_dl, device, logger)

        loss_table = build_train_table(eval_metrics, model=model, weights=self.weights)
        logger.info("\nEval loss metrics:\n%s", loss_table)

        eval_table = build_eval_table(factor_metrics)
        logger.info("\nEval factor metrics:\n%s", eval_table)

        xp.link.push_metrics({"expert/eval": eval_metrics})
        if factor_metrics:
            xp.link.push_metrics({"expert/factors": factor_metrics})
        return eval_metrics, factor_metrics


@hydra_main(config_path="conf", config_name="default", version_base="1.1")
def main(cfg):
    logger = get_logger("train_expert.log")
    xp = get_xp()
    logger.info(f"Exp signature: {xp.sig}")
    logger.info(repr(cfg))
    logger.info(f"Work dir: {os.getcwd()}")
    logger.info(f"Exec file: {__file__}")

    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("No GPU available, switching to CPU")
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"

    trainer = ExpertTrainer(cfg)

    train_dl, eval_dl = initialize_dataloaders(cfg, logger)

    if cfg.eval.eval_only:
        trainer.evaluate(cfg, logger, eval_dl, xp)
    else:
        trainer.train(cfg, logger, train_dl, eval_dl, xp)


if __name__ == "__main__":
    main()
