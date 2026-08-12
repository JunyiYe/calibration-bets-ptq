"""Rescue experiments: mixed precision and percentile calibration vs full W4A4.

For each (fold, seed) checkpoint of a config, evaluates
  - fp32 (reference),
  - full W4A4 abs-max (reproduced for pairing),
  - mixed: top-damage layer(s) at 8-bit, rest at 4-bit (--protect or
    --protect-auto, which reads the layer_sensitivity JSON),
  - W4A4 with percentile calibration (--percentiles).
Paired-by-day deltas vs fp32 per cell, aggregated across cells.

    python scripts/rescue_check.py --config configs/patchtst_sp500_vol_tuned.yaml \
        --protect-auto --percentiles 0.999,0.99
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from calibration_bets.config import load_config
from calibration_bets.experiment import build_calibration_loader, build_datasets
from calibration_bets.metrics.significance import daily_ic_frame, paired_delta_tests
from calibration_bets.models.factory import build_model
from calibration_bets.quantization.rtn import fake_static_int8, fake_static_mixed
from calibration_bets.training.loop import predict
from calibration_bets.utils.seed import set_seed


def parse_list(v: str, cast=str) -> list:
    return [cast(p) for p in str(v).split(",") if p != ""]


def fold_cfg(base: dict, test_year: int, train_years: int | None = None) -> dict:
    cfg = copy.deepcopy(base)
    cfg["data"]["train_end"] = f"{test_year - 2}-12-31"
    cfg["data"]["val_end"] = f"{test_year - 1}-12-31"
    cfg["data"]["test_end"] = f"{test_year}-12-31"
    if train_years is not None:
        # must match the training-time window: normalization stats come from
        # the train slice even when only val/test splits are built
        cfg["data"]["train_start"] = f"{test_year - 2 - train_years}-12-31"
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Mixed-precision / percentile rescue check.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--folds", default="2021,2022,2023,2024,2025")
    parser.add_argument("--seeds", default="7,8,9,10,11")
    parser.add_argument("--sweep-dir", default="outputs/sweep")
    parser.add_argument("--protect", default=None, help="Comma layer names for mixed precision.")
    parser.add_argument("--protect-auto", action="store_true",
                        help="Protect the top only-one layer from the layer_sensitivity JSON.")
    parser.add_argument("--percentiles", default="0.999,0.99")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--policy", default="recent")
    parser.add_argument("--train-years", type=int, default=None,
                        help="Fixed training-window length in years; must match the sweep that trained the checkpoints.")
    parser.add_argument("--output-dir", default="outputs/mechanism")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--dump-daily", default=None,
                        help="Directory for per-cell daily IC CSVs (date-indexed, one column per method).")
    parser.add_argument("--calib-split", default="val", choices=("val", "test"),
                        help="Split to draw calibration windows from. 'test' is the "
                             "ORACLE setting (look-ahead; diagnostic only, not deployable).")
    parser.add_argument("--eval-split", default="test", choices=("val", "test"),
                        help="Split to EVALUATE on. 'val' scores the sweep on the "
                             "validation year: the deployable-proxy diagnostic "
                             "(everything a deployer sees before the test year).")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    stem = Path(args.config).stem
    model_key = stem

    protect: list[str] = parse_list(args.protect) if args.protect else []
    if args.protect_auto:
        sens_path = Path("outputs/mechanism") / f"{model_key}_fold2024_seed7_layer_sensitivity.json"
        sens = json.loads(sens_path.read_text(encoding="utf-8"))
        top = max(sens["layers"].items(), key=lambda kv: kv[1]["only_one"]["delta_ic_mean"])
        protect = [top[0]]
        print(f"[PROTECT] auto: {protect} (only-one delta {top[1]['only_one']['delta_ic_mean']:+.4f})", flush=True)
    percentiles = parse_list(args.percentiles, float) if args.percentiles else []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hac_lag = max(0, int(base_cfg["task"].get("forward_period", 1)) - 1)
    per_cell: dict[str, list[float]] = {}
    cell_rows: list[dict] = []

    for fold in parse_list(args.folds, int):
        cfg = fold_cfg(base_cfg, fold, train_years=args.train_years)
        data = build_datasets(cfg, splits=("val", "test"))
        eval_ds = data.splits[args.eval_split]
        loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
        calibration_cfg = cfg.get("ptq", {}).get("calibration", {})
        for seed in parse_list(args.seeds, int):
            ckpt = Path(args.sweep_dir) / f"{stem}_fold{fold}_seed{seed}_model.pt"
            if not ckpt.exists():
                print(f"[SKIP] missing {ckpt}", flush=True)
                continue
            set_seed(seed)
            cfg["seed"] = seed
            model = build_model(cfg["model"], cfg["task"], num_features=len(data.feature_columns))
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            cal_inputs = [b[0] for b in build_calibration_loader(
                data.splits[args.calib_split], policy=args.policy,
                calibration_cfg=calibration_cfg,
                seed=seed, batch_size=int(args.batch_size))]

            preds: dict[str, np.ndarray] = {}
            target_true, preds["fp32"] = predict(model, loader, device)
            _, preds["w4a4_absmax"] = predict(
                fake_static_int8(model, cal_inputs, num_bits=args.bits), loader, device)
            if protect:
                _, preds["mixed_protect"] = predict(
                    fake_static_mixed(model, cal_inputs, protect=set(protect), low_bits=args.bits), loader, device)
            for p in percentiles:
                _, preds[f"w4a4_p{p:g}"] = predict(
                    fake_static_int8(model, cal_inputs, num_bits=args.bits, percentile=p), loader, device)

            frame = daily_ic_frame(eval_ds.meta, target_true, preds)
            if args.dump_daily:
                dump_dir = Path(args.dump_daily)
                dump_dir.mkdir(parents=True, exist_ok=True)
                frame.to_csv(dump_dir / f"{model_key}_fold{fold}_seed{seed}_daily_ic.csv")
            tests = paired_delta_tests(frame, baseline="fp32", lag=hac_lag)
            parts = []
            for k, t in tests.items():
                per_cell.setdefault(k, []).append(t["delta_ic_mean"])
                cell_rows.append({"arch": model_key, "fold": fold, "seed": seed,
                                  "method": k, "delta": round(t["delta_ic_mean"], 6)})
                parts.append(f"{k}={t['delta_ic_mean']:+.4f}")
            print(f"[CELL] fold{fold} seed{seed}: " + "  ".join(parts), flush=True)

    summary = {
        "config": str(args.config),
        "protect": protect,
        "percentiles": percentiles,
        "eval_split": args.eval_split,
        "cells": len(next(iter(per_cell.values()), [])),
        "delta_ic": {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
            for k, v in per_cell.items()
        },
    }
    fold_tag = str(args.folds).replace(",", "_")
    oracle_tag = "" if args.calib_split == "val" else "_oracle"
    eval_tag = "" if args.eval_split == "test" else "_evalval"
    out = Path(args.output_dir) / f"{model_key}_rescue_folds{fold_tag}{oracle_tag}{eval_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if cell_rows:
        import csv
        cells_out = out.with_name(out.stem.replace("_rescue_", "_cells_") + ".csv")
        with cells_out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["arch", "fold", "seed", "method", "delta"])
            writer.writeheader()
            writer.writerows(cell_rows)
        print(f"[OK] Wrote {cells_out}")
    print(json.dumps(summary["delta_ic"], indent=2, sort_keys=True))
    print(f"[OK] Wrote {out}")


if __name__ == "__main__":
    main()
