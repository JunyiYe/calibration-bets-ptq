#!/usr/bin/env bash
# Full pipeline, sequential. On a cluster, parallelize the per-architecture
# loops (one job each); every step is resumable.
set -euo pipefail
cd "$(dirname "$0")"

# Frozen configs: hyperparameters were selected once on a development
# pseudo-fold (train 2009-2015, validate 2016) and never touched again.
ARCHS=(dlinear patchtst itransformer tsmixer transformer timemixer segrnn)
FOLDS=2018,2019,2020,2021,2022,2023,2024,2025
SEEDS=1,2,3,4,5,6,7,8,9,10

# 1. Data: the exact panel used in the paper ships with the repo. This step
#    only runs if you delete it to rebuild from scratch.
if [ ! -f data/sp500_returns.csv ]; then
  python scripts/download_yfinance_panel.py \
    --tickers configs/sp500_tickers.txt \
    --output data/sp500_returns.csv \
    --start 2008-06-01 --end 2026-01-01
  python scripts/backfill_missing_tickers.py \
    --panel data/sp500_returns.csv --tickers BK,CTRA,DAY,HOLX,MMC
fi

# 2. Training + PTQ evaluation (7 archs x 8 folds x 10 seeds, fixed 7-year window)
for arch in "${ARCHS[@]}"; do
  python scripts/run_sweep.py \
    --config "configs/${arch}.yaml" \
    --test-years "$FOLDS" --seeds "$SEEDS" \
    --train-years 7 \
    --output-dir outputs/sweep
done

# 3a. Percentile-calibrated W4A4 (reuses step-2 checkpoints; calibration
#     windows sampled uniformly from the validation year, same as the sweep)
for arch in "${ARCHS[@]}"; do
  python scripts/rescue_check.py \
    --config "configs/${arch}.yaml" \
    --folds "$FOLDS" --seeds "$SEEDS" \
    --percentiles 0.999,0.99 \
    --policy random --train-years 7 \
    --sweep-dir outputs/sweep --output-dir outputs/mechanism
done

# 3b. HAR / persistence baselines (same folds, same fixed window)
python scripts/har_baseline.py \
  --data data/sp500_returns.csv --folds "$FOLDS" --train-years 7

# 4. Aggregate results (console + outputs/summary.csv)
python scripts/summarize_results.py
