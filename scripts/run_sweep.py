"""Rolling-fold x multi-seed sweep for one base config.

Expanding-window folds: for test year Y, train uses data through Y-2,
validation is year Y-1, test is year Y. Each (fold, seed) trains its own
checkpoint and writes label-tagged metrics/preds via calibration_bets.runner.

Usage (inside the cluster container):
    python scripts/run_sweep.py --config configs/tsmixer_sp500_vol_opt.yaml \
        --test-years 2021,2022,2023,2024,2025 --seeds 7,8,9,10,11
Resumable: existing checkpoints are reused, and --skip-existing skips
(fold, seed) cells whose metrics JSON is already present.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from calibration_bets.config import load_config
from calibration_bets.experiment import parse_range
from calibration_bets.runner import run_experiment


def parse_ints(v: str) -> list[int]:
    return [int(p) for p in str(v).split(",") if p != ""]


def fold_config(base_cfg: dict, test_year: int, seed: int, train_years: int | None = None) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    cfg["data"]["train_end"] = f"{test_year - 2}-12-31"
    cfg["data"]["val_end"] = f"{test_year - 1}-12-31"
    cfg["data"]["test_end"] = f"{test_year}-12-31"
    if train_years is not None:
        # fixed training window: exactly train_years calendar years ending Y-2
        # (train_start is exclusive, so Y-2-train_years Dec 31 starts Jan 1)
        cfg["data"]["train_start"] = f"{test_year - 2 - train_years}-12-31"
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling-fold x seed sweep for one config.")
    parser.add_argument("--config", required=True, help="Base experiment YAML config.")
    parser.add_argument("--test-years", default="2021,2022,2023,2024,2025")
    parser.add_argument("--seeds", default="7,8,9,10,11")
    parser.add_argument("--train-years", type=int, default=None,
                        help="Fixed training-window length in years (default: expanding window).")
    parser.add_argument("--output-dir", default="outputs/sweep")
    parser.add_argument("--clip", default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    stem = Path(args.config).stem
    test_years = parse_ints(args.test_years)
    seeds = parse_ints(args.seeds)
    output_dir = Path(args.output_dir)
    clip_range = parse_range(args.clip)

    total = len(test_years) * len(seeds)
    done = 0
    failures: list[str] = []
    for test_year in test_years:
        for seed in seeds:
            done += 1
            label = f"{stem}_fold{test_year}_seed{seed}"
            metrics_path = output_dir / f"{label}_metrics.json"
            if args.skip_existing and metrics_path.exists():
                print(f"[SKIP {done}/{total}] {label} (metrics exist)", flush=True)
                continue
            print(f"[RUN {done}/{total}] {label}", flush=True)
            cfg = fold_config(base_cfg, test_year, seed, train_years=args.train_years)
            try:
                run_experiment(
                    cfg,
                    label=label,
                    output_dir=output_dir,
                    clip_range=clip_range,
                    config_name=str(args.config),
                )
            except Exception as exc:  # keep sweeping; report at the end
                print(f"[FAIL] {label}: {exc}", flush=True)
                failures.append(label)

    summary = {"config": str(args.config), "cells": total, "failures": failures}
    print(json.dumps(summary))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
