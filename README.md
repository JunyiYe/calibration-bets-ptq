# Calibration Bets on the Past: PTQ for Financial Time-Series Forecasting

Code to reproduce the experiments of *"Calibration Bets on the Past:
Post-Training Quantization for Financial Time-Series Forecasting"* (ICAIF '26,
accepted): a systematic study of activation calibration for post-training
quantization (PTQ) in cross-sectional volatility forecasting on the S&P 500,
covering seven architectures, eight walk-forward test years (2018-2025), and
560 trained models.

The pipeline: build the S&P 500 return panel, train each architecture on every
walk-forward fold and seed, evaluate each trained checkpoint under every PTQ
treatment (FP32 reference, dynamic INT8, static W8A8, weight-only W4, static
W4A4), re-evaluate W4A4 with percentile-calibrated activation ranges, and
compare against HAR / persistence baselines. Forecast skill is the daily
cross-sectional information coefficient (IC); quantization damage is the
paired difference IC(fp32) - IC(quantized) on identical stocks and dates.
Static W8A8, W4, and W4A4 use fake quantization with FP32
computation to measure predictive damage. These evaluations
do not measure low-precision hardware speed, memory, or energy savings.

## Layout

```text
configs/                 One frozen config per architecture + S&P 500 ticker list
scripts/                 Pipeline entrypoints (see below)
src/calibration_bets/          Library: data, models, training, PTQ, metrics
Time-Series-Library/     Vendored third-party model backbones (MIT,
                         thuml/Time-Series-Library, trimmed to the modules used)
data/                    S&P 500 returns panel (shipped; see step 1)
results/                 Released artifacts the paper references directly
outputs/                 Written by the pipeline (checkpoints, metrics, summaries)
```

Models are thin adapters around the official
[Time-Series-Library](https://github.com/thuml/Time-Series-Library)
implementations so PTQ coverage is uniform across architectures.
Hyperparameters were selected once on a development pseudo-fold (train
2009-2015, validate 2016) and are frozen in `configs/`.

## Setup

Python >= 3.10 and a CUDA GPU are assumed (paper runs used PyTorch 2.4.0 /
CUDA 12.4 in a container built from `docker://pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`).

```bash
pip install -r requirements.txt
pip install -e .
```

## Running the experiments

`bash reproduce.sh` runs steps 1–4 sequentially. On a cluster,
step 2 can be parallelized with one job per architecture.
Training skips completed cells; the percentile-rescue pass
reuses checkpoints but reruns its evaluations.

**1. Data.** The exact panel used in the paper ships with the repo:
`data/sp500_returns.csv` — daily simple returns from Yahoo Finance
adjusted closes for the 501 current S&P 500 constituents, June 2008 -
December 2025 (~2.1M asset-days). Nothing to run.
The panel is derived from Yahoo Finance data and is provided for research
reproducibility only; the underlying prices remain subject to Yahoo's terms.

To rebuild it from scratch instead (note: Yahoo Finance data can change over
time — delistings, adjustment revisions — so a fresh download may differ
marginally from the shipped panel):

```bash
python scripts/download_yfinance_panel.py \
  --tickers configs/sp500_tickers.txt \
  --output data/sp500_returns.csv \
  --start 2008-06-01 --end 2026-01-01
python scripts/backfill_missing_tickers.py \
  --panel data/sp500_returns.csv --tickers BK,CTRA,DAY,HOLX,MMC
```

**2. Training + PTQ evaluation.** Walk-forward protocol with a fixed 7-year
training window: for test year Y, train on [Y-8, Y-2], validate on Y-1
(early stopping), test on Y. Eight folds (2018-2025) x 10 seeds x 7
architectures = 560 training runs. Each (fold, seed) cell trains a checkpoint,
then evaluates it under every PTQ treatment; static activation ranges are
calibrated on 1,024 windows sampled uniformly from the validation year and
held fixed through the test year. Per cell, the runner writes the checkpoint,
predictions, and a metrics JSON with per-treatment daily IC. Models are small
(1.4k-100k parameters), so a single cell takes minutes on one GPU.

```bash
ARCHS="dlinear patchtst itransformer tsmixer transformer timemixer segrnn"
for arch in $ARCHS; do
  python scripts/run_sweep.py \
    --config configs/${arch}.yaml \
    --test-years 2018,2019,2020,2021,2022,2023,2024,2025 \
    --seeds 1,2,3,4,5,6,7,8,9,10 \
    --train-years 7 \
    --output-dir outputs/sweep
done
```

**3. Percentile-calibrated W4A4 + classical baselines.** The rescue pass
reuses the step-2 checkpoints (no retraining) and re-evaluates static W4A4
with activation ranges set by the p99.9 / p99 percentile of the calibration
sample instead of abs-max, paired by day against each cell's own FP32
forecasts. The HAR and persistence baselines run on the same folds:

```bash
for arch in $ARCHS; do
  python scripts/rescue_check.py \
    --config configs/${arch}.yaml \
    --folds 2018,2019,2020,2021,2022,2023,2024,2025 \
    --seeds 1,2,3,4,5,6,7,8,9,10 \
    --percentiles 0.999,0.99 \
    --policy random --train-years 7 \
    --sweep-dir outputs/sweep --output-dir outputs/mechanism
done

python scripts/har_baseline.py \
  --data data/sp500_returns.csv \
  --folds 2018,2019,2020,2021,2022,2023,2024,2025 --train-years 7
```

**4. Aggregate.** Summarizes the per-cell metrics into the paper's main
numbers (per architecture: FP32 IC and damage per treatment, mean +/- s.d.
across the 80 cells), printed to the console and written to
`outputs/summary.csv`:

```bash
python scripts/summarize_results.py
```

## Released artifacts

`results/` holds the two artifact sets the paper says accompany the released
code:

- `results/layer_sensitivity/` — full per-layer sensitivity maps (49 JSONs:
  7 cells per architecture). For each layer: the damage when only that layer
  is quantized to W4A4 and when every layer but that one is quantized.
- `results/clip_sweep/clip_sweep_cells.csv` — per-cell activation-range sweep
  behind the per-year range curves: Delta daily IC vs. FP32 for every
  (architecture, fold, seed) cell at each range setting (p90 ... p99.99,
  abs-max), 4,480 rows.

## License

MIT (see LICENSE).
