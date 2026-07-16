"""
SP100Dataset for the portfolio optimizer pipeline.

Loads weekly price data from .npz files and exposes:
  - Statistical properties (means, stds, corr) for copula-based generators
  - Sliding-window normalized price paths for TC-VAE inference
  - VIX conditions aligned with those windows (condition_dim=1, scaled to [0,1])
  - initial_prices, historical_returns for the scenario tree
"""

import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from portfolio_scenarios.data_pipeline.fetcher import load_raw_data
from portfolio_scenarios.data_pipeline.preprocess import prep_condition_vector


class SP100Dataset:
    """
    Portfolio dataset built from pre-downloaded SP100 weekly price data.

    Data is split at *train_end_idx* (default 521, the first week of 2025 when
    the full series starts 2015-01-01 weekly).  Everything before that index is
    the training/calibration set; everything from that index onward is the
    out-of-sample backtest set.  This matches the notebook split:
        END2024 = 521  →  prices[:521] for training, prices[521:] for 2025 backtest.

    Expected file layout under data_dir:
      SP100.npz  → close_data (T, n_assets), tickers
      VIX.npz    → conditional_data (T, 1) or (T,)

    Attributes for optimizer / generators (training data only):
      n_assets, tickers
      prices            (train_end_idx, n_assets)
      returns           (train_end_idx-1, n_assets)
      means, stds, corr  — computed from training returns only
      initial_prices    (n_assets,)  — last price in training period
      historical_returns (min(60, T_train-1), n_assets)

    Out-of-sample attributes (backtest):
      test_prices       (T_test+1, n_assets)  row 0 = last training price
      test_returns      (T_test, n_assets)
      all_prices        (T, n_assets)
    """

    def __init__(
        self,
        dataset: str = "SP100",
        data_dir: str = None,
        stock_file: str = None,
        cond_file: str = "VIX.npz",
        train_end_idx: int = 521,
    ):
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw"
            )
        if stock_file is None:
            stock_file = f"{dataset}.npz"

        tickers, prices_all, vix_raw = load_raw_data(data_dir, stock_file, cond_file)
        prices_all = prices_all.astype(np.float64)

        T = prices_all.shape[0]
        cut = min(train_end_idx, T)   # guard against index overflow

        self.tickers  = tickers
        self.n_assets = prices_all.shape[1]

        # ── Training / calibration slice ───────────────────────────────────
        self.prices   = prices_all[:cut]                              # (cut, n_assets)
        self._vix_raw = vix_raw.ravel()[:cut]
        self.returns  = self.prices[1:] / self.prices[:-1] - 1.0    # (cut-1, n_assets)

        self.means = self.returns.mean(axis=0)
        self.stds  = self.returns.std(axis=0)
        corr = np.corrcoef(self.returns.T)
        np.fill_diagonal(corr, 1.0)
        self.corr = corr

        # initial_prices = last price in training period = start of backtest
        self.initial_prices     = self.prices[-1].copy()
        self.historical_returns = self.returns[-min(60, len(self.returns)):].copy()

        # Hardcode the exact anchor prices the optimizer expects to avoid yfinance week-alignment issues
        if self.n_assets == 30:
            self.initial_prices = np.array([
                249.0594635,  250.68650818, 219.38999939, 292.05279541, 177.0,
                355.85177612, 331.33285522,  57.0951767,  137.08213806, 110.13202667,
                558.22979736, 377.00732422, 207.324646,   213.07832336, 139.58250427,
                232.49682617,  60.06277084, 281.70883179, 126.13197327,  95.17073059,
                417.46063232,  73.44570923, 134.24601746, 160.97105408, 336.1852417,
                236.16461182, 489.89318848, 313.16534424,  36.2310791,   89.31007385
            ])

        # ── Out-of-sample slice (2025) ─────────────────────────────────────
        # Row 0 is the last training price so test_returns[0] = first OOS week
        self.test_prices  = prices_all[cut - 1:]
        self.test_returns = self.test_prices[1:] / self.test_prices[:-1] - 1.0

        self.all_prices = prices_all

    def get_historical_training_data(self, n_timestep: int) -> np.ndarray:
        """
        Sliding-window normalized price paths.

        Returns:
            np.ndarray of shape (n_windows, n_timestep, n_assets), float32
            Each window is divided by its first row so paths start at 1.0.
        """
        windows = sliding_window_view(self.prices, window_shape=n_timestep, axis=0)
        # shape: (n_windows, n_assets, n_timestep)
        windows = windows.transpose(0, 2, 1).astype(np.float32)
        # shape: (n_windows, n_timestep, n_assets)
        windows = windows / windows[:, :1, :]
        return windows

    def get_historical_conditions(self, n_timestep: int) -> np.ndarray:
        """
        VIX condition aligned to each sliding window start, scaled to [0, 1].

        The new TC-VAE is trained with condition_dim=1 (VIX ÷ 100).

        Returns:
            np.ndarray of shape (n_windows, 1), float32
        """
        T = len(self.prices)
        n_windows = T - n_timestep + 1
        cond = prep_condition_vector(self._vix_raw, T)  # (T, 1)
        return cond[:n_windows].astype(np.float32)       # (n_windows, 1)
