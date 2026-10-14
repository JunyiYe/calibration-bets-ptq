# Calibration Bets on the Past

**Post-Training Quantization for Financial Time-Series Forecasting**
Junyi Ye, Ivy Gateri Wanjiku · accepted to ICAIF '26 · [paper PDF](paper/Calibration-Bets_ICAIF26.pdf)

This repo contains the code and data behind the paper. It studies how to
quantize trained volatility forecasters to low precision without retraining,
and shows that the choice of *activation calibration* is what decides whether
4-bit deployment works in financial forecasting.

## Why this paper

Financial forecasting models are usually developed in full precision, but
production deployment often needs low-precision inference to cut memory and
compute. Post-training quantization (PTQ) does this without retraining. Its
parts are not equally risky: weights can be quantized from the trained
parameters alone, but activation ranges have to be estimated from historical
data before deployment and then stay fixed on future data. In finance, that
future can look very different from the calibration period.

We test this on cross-sectional volatility forecasting for the S&P 500: seven
neural architectures, eight walk-forward test years (2018-2025), and 560
trained models. Each quantized model is compared against its own full-precision
checkpoint, so the only thing that changes is the quantization.

What we find:

- **At 8 bits, calibration barely matters.** 8-bit and weight-only 4-bit
  quantization cost little for most architectures.
- **At 4 bits, it is the main factor.** With default abs-max calibration,
  static 4-bit weights and activations (W4A4) remove 11-62% of the
  full-precision information coefficient (IC) in the affected architectures.
- **Much of that loss is recoverable.** Percentile calibration recovers 53-94%
  of it in the four most affected architectures, though recurrent and
  multi-scale models stay sensitive.
- **The best range changes over time.** Narrow ranges lose part of their
  advantage when test-period market dispersion exceeds what the calibration
  window saw.

So activation calibration should be treated as a deployment decision. When
damage remains, 8-bit activations or weight-only 4-bit are the safer choices.

## Quick start

```bash
pip install -r requirements.txt
pip install -e .
bash reproduce.sh          # trains every model, evaluates all PTQ treatments, aggregates
```

This reproduces the paper's main results end to end. It needs a CUDA GPU for
training. The models are small (1.4k-100k parameters), so one run takes
minutes, but the full sweep is 560 runs. Every step skips work that is already
done, so it is safe to stop and resume. The paper runs used PyTorch 2.4.0 with
CUDA 12.4.

To see the headline numbers once the sweep has finished:

```bash
python scripts/summarize_results.py     # prints per-architecture IC and damage, writes outputs/summary.csv
```

## What's inside

```text
src/calibration_bets/  Library: data, models, training, PTQ, metrics
scripts/               Pipeline entrypoints (training sweep, percentile rescue, HAR baseline, summary)
configs/               One frozen config per architecture, plus the S&P 500 ticker list
Time-Series-Library/   Vendored model backbones (MIT, thuml/Time-Series-Library, trimmed)
data/                  Daily S&P 500 returns panel, exactly what the paper used
results/               The paper's released artifacts (see below)
paper/                 Paper PDF
```

`outputs/` is created as you run the pipeline (checkpoints, per-cell metrics,
summaries).

## Released results

`results/` holds the artifacts the paper refers to directly:

- `results/layer_sensitivity/`: per-layer sensitivity maps, one JSON per
  architecture and cell (49 files). For each layer, the damage when only that
  layer is quantized to W4A4 and when every layer except that one is.
- `results/clip_sweep/clip_sweep_cells.csv`: the activation-range sweep behind
  the per-year range curves. Change in daily IC versus FP32 for every
  (architecture, fold, seed) cell at each range setting (p90 to p99.99, abs-max).

## Reproducing a specific result

`reproduce.sh` lists every step as a runnable command, in the order the tables
need them:

1. **Training and PTQ evaluation** (`scripts/run_sweep.py`). Walk-forward with a
   fixed 7-year training window: for test year Y, train on Y-8 to Y-2, validate
   on Y-1, test on Y. Each (fold, seed) cell is evaluated under FP32, dynamic
   INT8, static W8A8, weight-only W4 and static W4A4. Static ranges are
   calibrated on 1,024 windows sampled from the validation year and held fixed
   through the test year.
2. **Percentile-calibrated W4A4** (`scripts/rescue_check.py`). Reuses the
   step-1 checkpoints without retraining and re-evaluates W4A4 with p99.9 and
   p99 activation ranges.
3. **Baselines** (`scripts/har_baseline.py`). HAR and persistence on the same folds.
4. **Aggregation** (`scripts/summarize_results.py`).

Model hyperparameters are in `configs/` and were selected once on a development
pseudo-fold (train 2009-2015, validate 2016), then frozen.

## Notes

- `data/sp500_returns.csv` is the exact panel used in the paper: daily simple
  returns from Yahoo Finance adjusted closes for 501 current S&P 500
  constituents, June 2008 to December 2025. Yahoo revises its history over
  time, so a fresh download (`scripts/download_yfinance_panel.py`, then
  `scripts/backfill_missing_tickers.py`) will not match bit-for-bit.
- Quantization is simulated: static W8A8, W4 and W4A4 use fake quantization with
  FP32 arithmetic to measure predictive damage. The paper does not measure
  speed, memory or energy on low-precision hardware.
- Damage is the paired difference IC(fp32) - IC(quantized) on identical stocks
  and dates. Reported numbers are means across the 80 cells per architecture
  (8 folds x 10 seeds).

## Citation

```bibtex
@inproceedings{ye2026calibration,
  title     = {Calibration Bets on the Past: Post-Training Quantization for Financial Time-Series Forecasting},
  author    = {Ye, Junyi and Wanjiku, Ivy Gateri},
  booktitle = {Proceedings of the 7th ACM International Conference on AI in Finance (ICAIF '26)},
  year      = {2026}
}
```

## License

MIT (see LICENSE). Raw price data derive from Yahoo Finance and are provided
for research reproducibility only.
