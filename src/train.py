import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dora import get_xp, hydra_main

from .data import get_dataset, collate
from .models import ExpertModel
from .utils import get_logger, should_disable_tqdm


def _compute_expert_loss(model, batch, device, weights):
    embeddings = batch["embeddings"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    incoming = batch.get("incoming")
    outgoing = batch.get("outgoing")
    if incoming is not None:
        incoming = incoming.to(device, non_blocking=True)
    if outgoing is not None:
        outgoing = outgoing.to(device, non_blocking=True)

    outputs = model(embeddings, attention_mask, incoming, outgoing)
    batch_size = embeddings.size(0)

    sent_loss = F.mse_loss(outputs["reconstruction"], outputs["anchor"], reduction="sum") / max(batch_size, 1)

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

    total_loss = (
        weights["sent"] * sent_loss
        + weights["token"] * token_loss
        + weights["entropy"] * entropy_loss
        + weights["overlap"] * overlap_loss
        + weights["diversity"] * diversity_loss
        + weights["balance"] * balance_loss
    )

    metrics = {
        "total": float(total_loss.detach()),
        "sent": float(sent_loss.detach()),
        "token": float(token_loss.detach()),
        "entropy": float(entropy_loss.detach()),
        "overlap": float(overlap_loss.detach()),
        "diversity": float(diversity_loss.detach()),
        "balance": float(balance_loss.detach()),
    }

    return total_loss, metrics


def _update_metrics(collector, counts, metrics):
    for name, value in metrics.items():
        collector[name] += value
        counts[name] += 1


def _finalize_metrics(collector, counts):
    results = {}
    for name, value in collector.items():
        denom = counts.get(name, 0)
        results[name] = value / denom if denom else 0.0
    return results


def _format_metrics(metrics):
    return ", ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))


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


def _run_expert_epoch(
    model,
    loader,
    device,
    weights,
    *,
    train=False,
    optimizer=None,
    grad_clip=0.0,
    desc="",
    disable_progress=False,
):
    collector = defaultdict(float)
    counts = defaultdict(int)

    if train and optimizer is None:
        raise ValueError("Optimizer must be provided when train=True.")

    model.train() if train else model.eval()
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for batch in tqdm(loader, desc=desc, disable=disable_progress):
            if train:
                optimizer.zero_grad(set_to_none=True)
            total_loss, metrics = _compute_expert_loss(model, batch, device, weights)
            if train:
                total_loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            _update_metrics(collector, counts, metrics)

    return _finalize_metrics(collector, counts)


def _initialize_dataloaders(cfg, logger):
    train_ds, _ = get_dataset(
        name=cfg.data.train.dataset,
        subset=cfg.data.train.subset,
        rebuild=cfg.data.rebuild_ds,
        shuffle=cfg.data.train.shuffle,
    )
    eval_shuffle = bool(cfg.data.eval.shuffle)
    if eval_shuffle:
        logger.warning("Disabling shuffle for expert evaluation loader to preserve ordering.")
        eval_shuffle = False

    eval_ds, _ = get_dataset(
        split="validation",
        name=cfg.data.eval.dataset,
        subset=cfg.data.eval.subset,
        rebuild=cfg.data.rebuild_ds,
        shuffle=eval_shuffle,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.data.train.batch_size,
        collate_fn=collate,
        num_workers=cfg.data.train.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.data.train.num_workers > 0),
        shuffle=cfg.data.train.shuffle,
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=cfg.data.eval.batch_size,
        collate_fn=collate,
        num_workers=cfg.data.eval.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.data.eval.num_workers > 0),
        shuffle=eval_shuffle,
    )
    return train_dl, eval_dl


def _save_checkpoint(model, path, logger):
    torch.save(model.state_dict(), path)
    logger.info(f"Saved ExpertModel checkpoint to {path}")


def _load_checkpoint(model, path, device, logger):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expert checkpoint not found at {path}")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    logger.info(f"Loaded ExpertModel checkpoint from {path}")


def train_expert(cfg, logger, train_dl, eval_dl, xp):
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
    weights = _prepare_expert_weights(cfg)
    disable_progress = should_disable_tqdm()
    disable_eval_progress = should_disable_tqdm(metrics_only=True)

    checkpoint_path = "expert_model.pth"
    best_eval = float("inf")

    for epoch in range(cfg.train.epochs):
        train_metrics = _run_expert_epoch(
            model,
            train_dl,
            device,
            weights,
            train=True,
            optimizer=optimizer,
            grad_clip=grad_clip,
            desc=f"Expert Train {epoch + 1}",
            disable_progress=disable_progress,
        )
        logger.info(f"Epoch {epoch + 1} train: {_format_metrics(train_metrics)}")
        xp.link.push_metrics({f"expert/train/{epoch + 1}": train_metrics})

        eval_metrics = _run_expert_epoch(
            model,
            eval_dl,
            device,
            weights,
            train=False,
            desc=f"Expert Eval {epoch + 1}",
            disable_progress=disable_eval_progress,
        )
        logger.info(f"Epoch {epoch + 1} eval: {_format_metrics(eval_metrics)}")
        xp.link.push_metrics({f"expert/eval/{epoch + 1}": eval_metrics})

        current_eval = eval_metrics.get("total")
        if current_eval is not None and current_eval < best_eval:
            best_eval = current_eval
            _save_checkpoint(model, checkpoint_path, logger)

    if best_eval == float("inf"):
        _save_checkpoint(model, checkpoint_path, logger)

    return model


def evaluate_expert(cfg, logger, eval_dl, xp):
    device = cfg.device
    model = ExpertModel(cfg.model).to(device)
    checkpoint_path = "expert_model.pth"
    _load_checkpoint(model, checkpoint_path, device, logger)

    weights = _prepare_expert_weights(cfg)
    disable_progress = should_disable_tqdm(metrics_only=True)

    metrics = _run_expert_epoch(
        model,
        eval_dl,
        device,
        weights,
        train=False,
        desc="Expert Evaluation",
        disable_progress=disable_progress,
    )
    logger.info(f"Expert evaluation: {_format_metrics(metrics)}")
    xp.link.push_metrics({"expert/eval": metrics})
    return metrics


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

    train_dl, eval_dl = _initialize_dataloaders(cfg, logger)

    if cfg.eval.eval_only:
        evaluate_expert(cfg, logger, eval_dl, xp)
    else:
        train_expert(cfg, logger, train_dl, eval_dl, xp)


if __name__ == "__main__":
    main()
