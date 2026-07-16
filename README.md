# TC-VAE Scenario Generation

A **scenario-tree generator for two-stage stochastic portfolio optimization**,
built as a fork of [Time-Causal VAE](https://github.com/justinhou95/TimeCausalVAE).
It trains a conditional Time-Causal VAE on rolling windows of weekly equity
prices, samples forward price paths, couples them with an R-vine copula, and
exports discrete **scenario trees** in the CSV format consumed by the companion
[Stochastic-Portfolio-Optimizer](https://github.com/Ahkylez/Stochastic-Portfolio-Optimizer).

> This repository **adds** a portfolio layer on top of Time-Causal VAE and leaves
> the upstream model code untouched. See [Upstream](#built-on-time-causal-vae) and
> [`NOTICE`](NOTICE) for attribution.

---

## What this fork adds

Everything new lives in one place — the upstream `src/tsvae/` package is
unmodified:

| Path | Role |
|------|------|
| `portfolio_scenarios/scenario_generation/` | `DirectTCVAE` (samples the VAE) → `TCVAEVineScenarioTree` (R-vine copula + scenario-tree builder) → CSV exporters |
| `portfolio_scenarios/data_pipeline/` | Price/VIX loading, conditioning, windowed datasets for training |
| `notebooks/rolling_backtest_train.ipynb` | The paper pipeline: rolling-window retraining + scenario-tree export |

Importing `portfolio_scenarios` puts `src/` on `sys.path`, so the new code can
`import tsvae` and `import portfolio_scenarios.*` side by side without any
install step.

---

## How it works

For each weekly rebalancing step of a rolling backtest:

1. **Train** a conditional Time-Causal VAE (`BetaCVAE`, `CLSTMRes` encoder/decoder,
   `RealNVP` prior) on the trailing window of weekly returns, conditioned on VIX.
2. **Sample** a large pool of forward price paths from the trained VAE
   (`DirectTCVAE`).
3. **Couple & discretize** (`TCVAEVineScenarioTree`): fit an R-vine copula to the
   historical returns, simulate a stage-1 pool, reduce it with K-medoids, and
   solve an LP moment-matching problem for the node probabilities; then build
   conditional stage-2 nodes.
4. **Export** a two-stage scenario tree as CSV.

Run over 52 weekly steps × 10 random seeds, this produces the
`tcvae_week_NNN.csv` trees that drive the optimizer's dynamic backtest.

---

## Relationship to the optimizer

The two projects are decoupled by a **CSV contract** — no code dependency. Each
exported tree is a header-less CSV with `3 + 2Q` columns for `Q` assets:

| Columns | Meaning |
|---------|---------|
| `0` | recourse-node index `j` |
| `1` | `p_j` — probability of recourse node `j` |
| `2` | `p_je` — conditional probability of evaluate node `e` given `j` |
| `3 … 3+Q-1` | stage-1 (recourse) prices |
| `3+Q … 3+2Q-1` | stage-2 (evaluate) prices |

The training notebook writes these into the optimizer's
`data/scenarios/backtest/<seed>/` folders (path configurable via the
`OPTIMIZER_DIR` environment variable). See the optimizer repo's
`docs/scenario_csv_format.md` for the full spec.

---

## Setup (WSL / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs the CUDA build of PyTorch; training uses the GPU when
one is available under WSL2 and falls back to CPU otherwise. (For a CPU-only box,
install `torch` from the CPU index instead.)

Training data (`data/raw/DOW.npz`, `VIX.npz`) is included. Trained checkpoints and
generated results are gitignored (`results/`) — regenerate them by running the
notebook.

---

## Usage

Open `notebooks/rolling_backtest_train.ipynb`. The top **Configuration** cell sets
the rolling window, VAE architecture, scenario-tree sizes (`J` recourse × `E`
evaluate nodes), and `OPTIMIZER_DIR` (where the CSVs are written). Running the
notebook retrains per step and exports the scenario trees.

To point at a different optimizer checkout:

```bash
export OPTIMIZER_DIR=/path/to/Stochastic-Portfolio-Optimizer
```

---

## Built on Time-Causal VAE

The base model is **Time-Causal VAE**, included here unmodified:

- Repository: <https://github.com/justinhou95/TimeCausalVAE>
- Paper: *Time-Causal VAE: Robust Financial Time Series Generator*,
  [arXiv:2411.02947](https://arxiv.org/abs/2411.02947)

The upstream package (`src/tsvae/`), its evaluation suite (`src/evaluations/`),
example notebooks, configs, and shipped `trained_models/` come from that project.
For details on the VAE itself, its training pipeline, and the example datasets
(Black-Scholes, Heston, PDV, S&P500/VIX, toy 2D), refer to the upstream
repository.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE). Attribution for upstream and for the
added portfolio layer is recorded in [NOTICE](NOTICE).
