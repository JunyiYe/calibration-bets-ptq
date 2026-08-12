"""Programmatic experiment runner: train (or load) one model, evaluate all PTQ
methods, write metrics + predictions. `scripts/run_experiment.py` is a thin CLI
around this; `scripts/run_sweep.py` calls it per fold x seed."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from calibration_bets.backtest.long_short import long_short_backtest
from calibration_bets.data.windowing import TimeSeriesWindowDataset
from calibration_bets.experiment import build_calibration_loader, build_datasets
from calibration_bets.metrics.finance import summarize_finance
from calibration_bets.metrics.forecast import summarize_forecast
from calibration_bets.metrics.regime import summarize_by_abs_return_decile
from calibration_bets.metrics.significance import daily_ic_frame
from calibration_bets.models.factory import build_model
from calibration_bets.quantization.calibration import summarize_calibration_policies
from calibration_bets.quantization.noise import count_perturbed_parameters
from calibration_bets.quantization.ptq import quantize_model
from calibration_bets.training.loop import TrainConfig, TrainResult, predict, train_model
from calibration_bets.utils.device import resolve_device
from calibration_bets.utils.seed import set_seed


def run_experiment(
    cfg: dict[str, Any],
    label: str,
    output_dir: str | Path = "outputs",
    clip_range: tuple[float, float] | None = None,
    config_name: str | None = None,
) -> dict[str, Any]:
    """Run one full experiment for `cfg`; artifact filenames derive from `label`.

    Writes <output_dir>/<label>_model.pt (reused when present),
    <label>_metrics.json, and <label>_preds.npz. Returns the metrics dict.
    """
    output_dir = Path(output_dir)
    set_seed(int(cfg.get("seed", 7)))

    task_cfg = cfg["task"]
    data = build_datasets(cfg)
    feat_cols = data.feature_columns
    window_cfg = data.window_config
    train_ds = data.splits["train"]
    val_ds = data.splits["val"]
    test_ds = data.splits["test"]

    train_cfg = TrainConfig(**pick_keys(cfg.get("train", {}), TrainConfig.__annotations__.keys()))
    batch_size = int(cfg.get("train", {}).get("batch_size", train_cfg.batch_size))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    calibration_summary = build_calibration_summary(cfg, val_ds)
    if calibration_summary:
        print(f"[CALIBRATION] policies={list(calibration_summary.keys())}", flush=True)

    device = resolve_device(cfg.get("train", {}).get("device", "auto"))
    model = build_model(cfg["model"], task_cfg, num_features=len(feat_cols))

    checkpoint_path = output_dir / f"{label}_model.pt"
    if checkpoint_path.exists():
        print(f"[LOAD] Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        train_result = TrainResult(best_val_loss=float("nan"), epochs_ran=0)
    else:
        train_result = train_model(model, train_loader, val_loader, train_cfg, device)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[SAVE] Checkpoint saved to {checkpoint_path}")

    methods = cfg.get("ptq", {}).get("methods", ["fp32"])
    evaluation_cfg = cfg.get("evaluation", {})
    portfolio_cfg = evaluation_cfg.get("portfolio", {})
    regime_cfg = evaluation_cfg.get("regimes", {})
    naive_baselines = evaluate_naive_baselines(train_ds, val_ds, test_ds, portfolio_cfg, regime_cfg)

    results: dict[str, Any] = {
        "config": config_name if config_name is not None else label,
        "label": label,
        "git_commit": read_git_commit(),
        "quantization_coverage": count_perturbed_parameters(model),
        "seed": int(cfg.get("seed", 7)),
        "data_split": {
            "train_start": str(cfg["data"].get("train_start")),
            "train_end": str(cfg["data"].get("train_end")),
            "val_end": str(cfg["data"].get("val_end")),
            "test_end": str(cfg["data"].get("test_end")),
        },
        "train": {
            "best_val_loss": train_result.best_val_loss,
            "epochs_ran": train_result.epochs_ran,
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "test_windows": len(test_ds),
            "device": str(device),
            "target_transform": window_cfg.target_transform,
        },
        "calibration": calibration_summary,
        "baselines": naive_baselines,
        "methods": {},
    }

    calibration_cfg = cfg.get("ptq", {}).get("calibration", {})
    calibration_policies = list(calibration_cfg.get("policies", []))

    method_preds: dict[str, np.ndarray] = {}
    for method in methods:
        if method in ("static_int8", "static_int4"):
            if not calibration_policies:
                raise ValueError(f"{method} requires ptq.calibration.policies.")
            for policy in calibration_policies:
                method_label = f"{method}_{policy}"
                calibration_loader = build_calibration_loader(
                    val_ds,
                    policy=policy,
                    calibration_cfg=calibration_cfg,
                    seed=int(cfg.get("seed", 7)),
                    batch_size=batch_size,
                )
                qmodel = quantize_model(model, method, calibration_loader)
                results["methods"][method_label], method_preds[method_label] = evaluate_method(
                    qmodel,
                    test_loader,
                    test_ds,
                    device,
                    portfolio_cfg,
                    regime_cfg,
                    clip_range=clip_range,
                )
        else:
            # torch dynamic quantization only has CPU kernels; fake-quant
            # methods run fine on GPU (required for unrolled FakeQuantGRU speed)
            eval_device = torch.device("cpu") if method == "dynamic_int8" else device
            qmodel = quantize_model(model, method)
            results["methods"][method], method_preds[method] = evaluate_method(
                qmodel,
                test_loader,
                test_ds,
                eval_device,
                portfolio_cfg,
                regime_cfg,
            )

    output = output_dir / f"{label}_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[OK] Wrote metrics to {output}")

    # persist per-method test predictions so downstream analyses (paired
    # significance tests, noise-equivalence, plots) never need re-inference
    preds_path = output_dir / f"{label}_preds.npz"
    save_predictions(preds_path, test_ds, method_preds)
    print(f"[OK] Wrote predictions to {preds_path}")
    return results


def evaluate_naive_baselines(
    train_ds: TimeSeriesWindowDataset,
    val_ds: TimeSeriesWindowDataset,
    test_ds: TimeSeriesWindowDataset,
    portfolio_cfg: dict[str, Any],
    regime_cfg: dict[str, Any],
) -> dict[str, Any]:
    train_mean = float(train_ds.y.mean())
    val_mean = float(val_ds.y.mean())
    baselines = {
        "zero": np.zeros_like(test_ds.y),
        "train_mean": np.full_like(test_ds.y, train_mean),
        "val_mean": np.full_like(test_ds.y, val_mean),
    }
    result = {}
    for name, pred in baselines.items():
        result[name] = evaluate_arrays(test_ds.y, test_ds.raw_y, pred, test_ds, portfolio_cfg, regime_cfg)
    return result


def evaluate_arrays(
    target_true: np.ndarray,
    raw_true: np.ndarray,
    y_pred: np.ndarray,
    test_ds: TimeSeriesWindowDataset,
    portfolio_cfg: dict[str, Any],
    regime_cfg: dict[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, float] = {}
    metrics.update(summarize_forecast(target_true, y_pred))
    metrics.update(summarize_finance(raw_true, y_pred))
    # mean daily cross-sectional IC — the literature-standard signal metric;
    # the pooled `ic` above mixes days and overweights high-dispersion dates
    ic_days = daily_ic_frame(test_ds.meta, target_true, {"pred": y_pred})
    rank_ic_days = daily_ic_frame(test_ds.meta, target_true, {"pred": y_pred}, rank=True)
    metrics["daily_ic_mean"] = float(ic_days["pred"].mean()) if len(ic_days) else 0.0
    metrics["daily_ic_std"] = float(ic_days["pred"].std(ddof=1)) if len(ic_days) > 1 else 0.0
    metrics["daily_rank_ic_mean"] = float(rank_ic_days["pred"].mean()) if len(rank_ic_days) else 0.0
    metrics["n_days"] = float(len(ic_days))
    if portfolio_cfg.get("enabled", True):
        metrics.update(
            long_short_backtest(
                test_ds.meta,
                raw_true,
                y_pred,
                top_quantile=float(portfolio_cfg.get("top_quantile", 0.1)),
                bottom_quantile=float(portfolio_cfg.get("bottom_quantile", 0.1)),
                transaction_cost_bps=float(portfolio_cfg.get("transaction_cost_bps", 5.0)),
            )
        )
    regime_deciles = int(regime_cfg.get("volatility_deciles", 10))
    return {
        "overall": metrics,
        "abs_return_regimes": summarize_by_abs_return_decile(
            raw_true,
            y_pred,
            regime_deciles,
            forecast_true=target_true,
        ),
    }


def evaluate_method(
    model: torch.nn.Module,
    test_loader: DataLoader,
    test_ds: TimeSeriesWindowDataset,
    device: torch.device,
    portfolio_cfg: dict[str, Any],
    regime_cfg: dict[str, Any],
    clip_range: tuple[float, float] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    target_true, y_pred = predict(model, test_loader, device)
    if clip_range is not None:
        low, high = clip_range
        y_pred = np.clip(y_pred, low, high)
    return evaluate_arrays(target_true, test_ds.raw_y, y_pred, test_ds, portfolio_cfg, regime_cfg), y_pred


def build_calibration_summary(cfg: dict[str, Any], val_ds: TimeSeriesWindowDataset) -> dict[str, dict[str, float | str]]:
    calibration_cfg = cfg.get("ptq", {}).get("calibration", {})
    policies = calibration_cfg.get("policies", [])
    if not policies:
        return {}
    return summarize_calibration_policies(
        val_ds.meta,
        val_ds.raw_y,
        policies=list(policies),
        num_windows=int(calibration_cfg.get("num_windows", 1024)),
        volatility_window=int(calibration_cfg.get("volatility_window", 20)),
        seed=int(cfg.get("seed", 7)),
        inputs=val_ds.x,
    )


def save_predictions(
    path: Path,
    test_ds: TimeSeriesWindowDataset,
    method_preds: dict[str, np.ndarray],
) -> None:
    arrays: dict[str, np.ndarray] = {
        "target_true": test_ds.y.astype(np.float32),
        "raw_y": test_ds.raw_y.astype(np.float32),
        "date_ns": test_ds.meta["date"].to_numpy().astype("datetime64[ns]").astype(np.int64),
        "asset": test_ds.meta["asset"].to_numpy().astype(str),
    }
    for label, pred in method_preds.items():
        arrays[f"pred::{label}"] = np.asarray(pred, dtype=np.float32)
    np.savez_compressed(path, **arrays)


def read_git_commit(repo_root: str | Path = ".") -> str | None:
    """Current commit hash, read from .git directly (containers have no git binary)."""
    try:
        git_dir = Path(repo_root) / ".git"
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(None, 1)[1]
        ref_path = git_dir / ref
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
        packed = git_dir / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(ref) and not line.startswith("#"):
                    return line.split()[0]
    except OSError:
        pass
    return None


def pick_keys(source: dict[str, Any], keys: object) -> dict[str, Any]:
    allowed = set(keys)
    return {k: v for k, v in source.items() if k in allowed}
