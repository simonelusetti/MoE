"""Dora grid runners for MoE experiments driven by YAML sweep files."""

import itertools
from pathlib import Path

import treetable as tt
import yaml
from dora import Explorer


def load_yaml_sweep(config_filename: str):
    path = Path(__file__).resolve().parent / config_filename
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    baseline = data.get("baseline", {})
    sweep = data.get("sweep", {})

    if not isinstance(sweep, dict):
        raise ValueError("'sweep' must be a dictionary where each value is a list.")

    keys = list(sweep.keys())
    values = list(sweep.values())

    for key, val in zip(keys, values):
        if not isinstance(val, (list, tuple)):
            raise ValueError(f"Sweep entry '{key}' must be a list of values.")

    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return baseline, combinations


class BaseExplorer(Explorer):
    metric_names = ("precision", "recall", "f1")
    factor_prefix = "expert"

    def get_grid_metrics(self):
        return [
            tt.group(
                "best",
                [
                    tt.leaf("epoch", align=">"),
                    tt.leaf("factor"),
                    *(tt.leaf(name, ".4f") for name in self.metric_names),
                ],
            )
        ]

    def process_history(self, history):
        best_epoch, best_factor, best_metrics = summarize_best_factor(
            history, self.metric_names, self.factor_prefix
        )
        result = {
            "best": {
                "epoch": best_epoch,
                "factor": best_factor,
            }
        }
        result["best"].update(best_metrics)
        return result


class ExpertExplorer(BaseExplorer):
    factor_prefix = "expert"


def run_yaml_explorer(launcher, config_filename: str):
    baseline, combinations = load_yaml_sweep(config_filename)
    configured = launcher.bind(baseline) if baseline else launcher

    for overrides in combinations:
        configured(overrides)


@ExpertExplorer
def explorer(launcher):
    """Launch expert-model experiments defined in grid.yaml."""
    run_yaml_explorer(launcher, "grid.yaml")


def summarize_best_factor(history, metric_names, prefix: str):
    """Return (best_epoch, best_factor, metrics_dict) from Dora history."""

    def empty_metrics():
        return {name: None for name in metric_names}

    best_epoch = None
    best_factor = None
    best_metrics = empty_metrics()

    # Prefer explicit best metrics pushed at the end of training.
    for entry in history:
        if not isinstance(entry, dict):
            continue
        epoch_val = entry.get(f"{prefix}/best_epoch")
        factor_val = entry.get(f"{prefix}/best_factor")
        f1_val = entry.get(f"{prefix}/best_f1")
        if epoch_val is None or factor_val is None or f1_val is None:
            continue

        best_epoch = int(epoch_val)
        best_factor = str(factor_val)
        best_metrics["f1"] = float(f1_val)
        factor_payload = _lookup_factor_metrics(history, prefix, best_epoch, best_factor)
        for name in metric_names:
            if name in factor_payload:
                best_metrics[name] = factor_payload[name]
        return best_epoch, best_factor, best_metrics

    # Fallback: scan factor metrics to find the highest F1.
    best_f1 = float("-inf")
    for entry in history:
        if not isinstance(entry, dict):
            continue
        for path, payload in entry.items():
            if not isinstance(path, str) or not isinstance(payload, dict):
                continue
            if not path.startswith(f"{prefix}/factors/"):
                continue

            epoch = _extract_epoch_from_path(path)
            for factor, metrics in payload.items():
                if not isinstance(metrics, dict):
                    continue
                f1 = metrics.get("f1")
                if not isinstance(f1, (int, float)):
                    continue
                if f1 > best_f1:
                    best_f1 = float(f1)
                    best_epoch = epoch
                    best_factor = factor
                    best_metrics = {name: metrics.get(name) for name in metric_names}
    return best_epoch, best_factor, best_metrics


def _extract_epoch_from_path(path):
    """Best-effort extraction of epoch index from metric path."""
    segments = path.split("/")
    for segment in reversed(segments):
        if segment.isdigit():
            return int(segment)
    return None


def _lookup_factor_metrics(history, prefix, epoch, factor):
    """Find metrics for a specific epoch and factor inside the Dora history."""
    if epoch is None or factor is None:
        return {}
    target_path = f"{prefix}/factors/{epoch}"
    for entry in history:
        if not isinstance(entry, dict):
            continue
        for path, payload in entry.items():
            if path != target_path or not isinstance(payload, dict):
                continue
            metrics = payload.get(factor)
            if isinstance(metrics, dict):
                return metrics
    return {}
