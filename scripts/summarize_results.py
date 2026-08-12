"""Aggregate the experiment outputs into the paper's main-result numbers.

Reads
  outputs/sweep/<config>_fold<Y>_seed<S>_metrics.json  (scripts/run_sweep.py)
  outputs/mechanism/<arch>_rescue_folds*.json          (scripts/rescue_check.py)
  outputs/har_baseline.json                            (scripts/har_baseline.py)
and reports, per architecture: FP32 daily IC and the quantization damage
(Delta IC = IC_fp32 - IC_quantized) under each treatment, mean +/- s.d.
across fold-by-seed cells. Prints to stdout and writes outputs/summary.csv.

    python scripts/summarize_results.py
"""
from __future__ import annotations

import csv
import glob
import json
import statistics as st
from pathlib import Path

SWEEP = "outputs/sweep"
MECH = "outputs/mechanism"
HAR = "outputs/har_baseline.json"
OUT = Path("outputs/summary.csv")

ARCHS = {
    "dlinear": ("DLinear", "linear"),
    "patchtst": ("PatchTST", "attention"),
    "itransformer": ("iTransformer", "attention"),
    "tsmixer": ("TSMixer", "mixer"),
    "transformer": ("Transformer", "attention"),
    "timemixer": ("TimeMixer", "mixer"),
    "segrnn": ("SegRNN", "recurrent"),
}


def arch_stats(a: str) -> dict | None:
    fps, ddyn, d8, dw, d4 = [], [], [], [], []
    for f in glob.glob(f"{SWEEP}/{a}*_metrics.json"):
        me = json.load(open(f))["methods"]
        g = lambda k: me[k]["overall"]["daily_ic_mean"]
        fps.append(g("fp32"))
        ddyn.append(g("fp32") - g("dynamic_int8"))
        d8.append(g("fp32") - g("static_int8_random"))
        dw.append(g("fp32") - g("rtn_w4"))
        d4.append(g("fp32") - g("static_int4_random"))
    if not fps:
        return None
    groups = {"w4a4_p0.99": [], "w4a4_p0.999": []}  # (n, mean, sd) per rescue file
    for rf in glob.glob(f"{MECH}/{a}_rescue_folds*.json"):
        if "oracle" in rf:
            continue
        r = json.load(open(rf))
        for k in groups:
            groups[k].append((r["cells"], r["delta_ic"][k]["mean"], r["delta_ic"][k]["std"]))

    def pooled(parts):
        # combine per-file (n, mean, sd) into overall mean and sd (ddof=1)
        if not parts:
            return None, None
        n_tot = sum(n for n, _, _ in parts)
        mean = sum(n * m for n, m, _ in parts) / n_tot
        if n_tot < 2:  # single cell: a spread is undefined, not zero
            return mean, None
        ss = sum(s * s * (n - 1) + n * (m - mean) ** 2 for n, m, s in parts)
        return mean, (ss / (n_tot - 1)) ** 0.5

    p99, p99_sd = pooled(groups["w4a4_p0.99"])
    p999, p999_sd = pooled(groups["w4a4_p0.999"])
    return {
        "n_cells": len(fps),
        "fp32": st.mean(fps), "fp32_sd": stdev_or_none(fps),
        "d_dyn8": st.mean(ddyn), "d_w8a8": st.mean(d8), "d_w4": st.mean(dw),
        "d_w4a4_absmax": st.mean(d4), "d_w4a4_absmax_sd": stdev_or_none(d4),
        "d_w4a4_p999": p999, "d_w4a4_p999_sd": p999_sd,
        "d_w4a4_p99": p99, "d_w4a4_p99_sd": p99_sd,
    }


def stdev_or_none(xs: list[float]) -> float | None:
    """Sample s.d., or None for a single cell (a reduced run, e.g. one fold x one seed)."""
    return st.stdev(xs) if len(xs) > 1 else None


def fmt(mean, sd=None) -> str:
    if mean is None:
        return "--"
    return f"{mean:+.4f}" if sd is None else f"{mean:+.3f}±{sd:.3f}"


def main() -> None:
    rows = {a: s for a in ARCHS if (s := arch_stats(a)) is not None}
    if not rows:
        raise SystemExit(f"no metrics found under {SWEEP}/ — run scripts/run_sweep.py first")
    order = sorted(rows, key=lambda a: rows[a]["d_w4a4_absmax"])

    header = ["model", "family", "n_cells", "fp32_ic", "d_dyn8", "d_w8a8", "d_w4",
              "d_w4a4_absmax", "d_w4a4_p999", "d_w4a4_p99"]
    print(f"{'model':<13}{'family':<11}{'fp32 IC':>14}{'Δdyn8':>9}{'ΔW8A8':>9}{'ΔW4':>9}"
          f"{'Δabs-max':>15}{'Δp99.9':>15}{'Δp99':>15}")
    try:
        har = json.load(open(HAR))
        for k, label in (("persistence", "Persistence"), ("har", "HAR")):
            ic = st.mean(v[k]["daily_ic_mean"] for v in har["folds"].values())
            print(f"{label:<13}{'baseline':<11}{ic:>14.3f}")
    except FileNotFoundError:
        print(f"[WARN] {HAR} not found — run scripts/har_baseline.py for the baselines")

    csv_rows = []
    for a in order:
        r = rows[a]
        name, fam = ARCHS[a]
        print(f"{name:<13}{fam:<11}{fmt(r['fp32'], r['fp32_sd']):>14}"
              f"{fmt(r['d_dyn8']):>9}{fmt(r['d_w8a8']):>9}{fmt(r['d_w4']):>9}"
              f"{fmt(r['d_w4a4_absmax'], r['d_w4a4_absmax_sd']):>15}"
              f"{fmt(r['d_w4a4_p999'], r['d_w4a4_p999_sd']):>15}"
              f"{fmt(r['d_w4a4_p99'], r['d_w4a4_p99_sd']):>15}")
        csv_rows.append([name, fam, r["n_cells"], r["fp32"], r["d_dyn8"], r["d_w8a8"],
                         r["d_w4"], r["d_w4a4_absmax"], r["d_w4a4_p999"], r["d_w4a4_p99"]])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(csv_rows)
    print(f"\n[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
