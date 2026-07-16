"""
Scenario generators for the two-stage stochastic CVaR portfolio optimizer.

Hierarchy:
  CopulaModel         – base class for parametric copula generators
    GaussianCopula
    StudentTCopula
    ClaytonCopula
  DirectTCVAE         – PRIMARY: use the joint TC-VAE directly for scenario generation.
  GARCHVineCopula     – Full He & Zhang (2024) baseline:
                          AR(1)-[GARCH/GJRGARCH/EGARCH](1,1)-SkewStudent marginals,
                          full R-vine copula family set (incl. rotations),
                          K-means scenario reduction, moment-matching probabilities,
                          and sequential multi-stage GARCH state propagation.

All generators expose:
  .n_assets  (int)
  .sample(n) → np.ndarray of shape (n, n_assets)  [one-step returns]

GARCHVineCopula additionally exposes:
  .has_sequential_sampling = True
  .sample_with_probs(n, prev_returns=None) → (centroids (n,d), probs (n,))
"""

import warnings

import numpy as np
import torch
from scipy.stats import norm

# Suppress GARCH optimizer convergence warnings globally for this module.
# arch uses scipy SLSQP which emits ConvergenceWarning; these are handled
# by the model-selection fallback chain so there is no action needed.
warnings.filterwarnings("ignore", message=".*Iteration limit reached.*")
warnings.filterwarnings("ignore", message=".*Inequality constraints incompatible.*")
warnings.filterwarnings("ignore", message=".*optimizer returned code.*")


# ─── Parametric copula base ────────────────────────────────────────────────────

class CopulaModel:
    """
    Parametric copula base class.

    Returns are back-transformed via:
        r = mu + sigma * Phi^{-1}(U)
    where U are the uniform marginals produced by each copula.
    """

    def __init__(self, means, stds, corr):
        self.means = np.asarray(means, dtype=float)
        self.stds = np.asarray(stds, dtype=float)
        self.corr = np.asarray(corr, dtype=float)
        self.n_assets = len(means)

    def _u_to_returns(self, U):
        U = np.clip(U, 1e-7, 1 - 1e-7)
        Z = norm.ppf(U)
        return self.means + self.stds * Z

    def _chol(self):
        R = self.corr + 1e-6 * np.eye(self.n_assets)
        return np.linalg.cholesky(R)

    def sample(self, n):
        raise NotImplementedError


# ─── Parametric copula generators ─────────────────────────────────────────────

class GaussianCopula(CopulaModel):
    """
    Gaussian copula.

    1. Z_ind  ~ N(0, I)
    2. Z_corr = Z_ind @ L^T,  L = chol(corr)
    3. U      = Phi(Z_corr)
    4. r      = mu + sigma * Phi^{-1}(U)
    """

    def __init__(self, means, stds, corr):
        super().__init__(means, stds, corr)
        self.L = self._chol()

    def sample(self, n):
        Z = np.random.randn(n, self.n_assets) @ self.L.T
        U = norm.cdf(Z)
        return self._u_to_returns(U)


class StudentTCopula(CopulaModel):
    """
    Student-t copula — heavier joint tails than Gaussian.

    1. Z  ~ N(0, corr)  via Cholesky
    2. W  ~ chi^2(df)   scalar per draw
    3. T  = Z / sqrt(W/df)
    4. U  = F_t(T; df)  element-wise t-CDF
    5. r  = mu + sigma * Phi^{-1}(U)
    """

    def __init__(self, means, stds, corr, df=5):
        super().__init__(means, stds, corr)
        self.df = float(df)
        self.L = self._chol()

    def sample(self, n):
        from scipy.stats import t as t_dist

        Z = np.random.randn(n, self.n_assets) @ self.L.T
        W = np.random.chisquare(df=self.df, size=(n, 1))
        T = Z / np.sqrt(W / self.df)
        U = t_dist.cdf(T, df=self.df)
        return self._u_to_returns(U)


class ClaytonCopula(CopulaModel):
    """
    Clayton copula — asymmetric lower-tail dependence.

    Gamma-frailty sampling:
    1. W  ~ Gamma(1/theta, 1)
    2. V_j ~ Exp(1)  iid, j = 1..n_assets
    3. U_j = (1 + V_j / W)^{-1/theta}
    4. r   = mu + sigma * Phi^{-1}(U)
    """

    def __init__(self, means, stds, corr, theta=2.0):
        super().__init__(means, stds, corr)
        self.theta = max(float(theta), 1e-4)

    def sample(self, n):
        th = self.theta
        W = np.random.gamma(shape=1.0 / th, scale=1.0, size=(n, 1))
        V = np.random.exponential(scale=1.0, size=(n, self.n_assets))
        U = (1.0 + V / W) ** (-1.0 / th)
        return self._u_to_returns(U)


# ─── Direct TC-VAE generation ──────────────────────────────────────────────────

class DirectTCVAE:
    """
    Scenario generator that uses the joint TC-VAE directly — no copula layer.

    Sampling strategy:
      Rather than sampling from the RealNVP prior p(z) (which can suffer from
      mode collapse when the flow is underfit), we bootstrap latent codes from
      the *empirical posterior* q(z|x) computed over the training windows.
      Each sample draws a random historical window's posterior mean + noise:
          z = mu_i + eps * exp(0.5 * log_var_i),  eps ~ N(0, I)
      This keeps z in the region of latent space the decoder was trained on,
      producing realistic and diverse scenarios.

    Volatility matching (mean-preserving):
      Generated returns are standardised to match the historical per-asset mean
      and standard deviation:
          r_matched = (r - mean(r)) / std(r) * hist_std + hist_mean
      This prevents mean amplification when the decoder output variance is small
      (prior collapse artifact), which would otherwise cause one asset to dominate
      all scenarios.

    Two-stage joint sampling:
      Both stage-1 (week 1) and stage-2 (week 2) returns are decoded from the
      SAME latent z, preserving causal temporal structure — a crash-regime z
      produces correlated stress at both stages.
    """

    def __init__(self, tcvae_model, historical_data, historical_conditions):
        """
        Args:
          tcvae_model:           trained BetaCVAE model
          historical_data:       np.ndarray (N, n_timestep, n_assets) — normalised windows
          historical_conditions: np.ndarray (N, cond_dim) — VIX conditions
        """
        self.model = tcvae_model
        self.n_assets = tcvae_model.model_config.data_dim
        self.device = next(tcvae_model.parameters()).device
        self._all_conds = historical_conditions  # (N, cond_dim)

        # Store normalized windows for historical bootstrap fallback
        self._hist_windows = historical_data.astype(np.float32)  # (N, T, n_assets), P(t)/P(0)

        # Per-asset historical stats for stage-1 (week-1 returns)
        hist_first_ret  = historical_data[:, 1, :].astype(np.float64) - 1.0        # P(1)/P(0) - 1
        self._hist_mean = hist_first_ret.mean(axis=0).astype(np.float32)           # (n_assets,)
        self._hist_std  = hist_first_ret.std(axis=0).astype(np.float32)            # (n_assets,)

        # Per-asset historical stats for stage-2 (week-2 conditional returns P(2)/P(1)-1)
        hist_second_ret  = (historical_data[:, 2, :] / historical_data[:, 1, :]).astype(np.float64) - 1.0
        self._hist_mean2 = hist_second_ret.mean(axis=0).astype(np.float32)         # (n_assets,)
        self._hist_std2  = hist_second_ret.std(axis=0).astype(np.float32)          # (n_assets,)

        # Historical correlation matrix — used by Iman-Conover to correct the
        # decoder's cross-asset correlation structure after vol-matching.
        # The LSTM decoder maps one latent z to one market state, so all assets
        # tend to co-move in generated paths even when historically uncorrelated.
        # Iman-Conover reorders the vol-matched draws to match this target.
        C = np.corrcoef(hist_first_ret.T)                                           # (Q, Q)
        C = (C + C.T) / 2                                                           # symmetrise
        np.fill_diagonal(C, 1.0)
        # Regularise: shrink 10% toward identity to ensure positive-definiteness
        Q = C.shape[0]
        self._hist_corr = 0.9 * C + 0.1 * np.eye(Q)
        self._hist_corr_chol = np.linalg.cholesky(self._hist_corr).astype(np.float32)

        # Pre-encode historical windows — posterior means/vars used for sampling
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(historical_data, dtype=torch.float32).to(self.device)
            c = torch.tensor(historical_conditions, dtype=torch.float32).to(self.device)
            x_t = self.model.transform(x)
            enc = self.model.encoder(x_t, c)
            self._mu      = enc.embedding.cpu().numpy()       # (N, latent_dim)
            self._log_var = enc.log_covariance.cpu().numpy()  # (N, latent_dim)

    def _sample_z(self, n: int) -> torch.Tensor:
        """
        Sample z for scenario generation.

        Factorized architecture (FactorizedCLSTMRes encoder/decoder)
        -------------------------------------------------------------
        z = [ z_market | z_asset_1 | ... | z_asset_Q ]

        z_market  — bootstrapped from the empirical posterior q(z_market | x).
                    Captures the market regime (bull/bear/volatile) for each scenario.
        z_asset_k — sampled independently from N(0, I) for EACH asset k.
                    This independence is what breaks cross-asset correlation:
                    different assets get different idiosyncratic shocks even when
                    the market factor is the same.

        Non-factorized architecture (CLSTMRes)
        --------------------------------------
        Full z bootstrapped from the posterior — unchanged behaviour.
        """
        cfg = self.model.model_config
        dim_m = getattr(cfg, "latent_dim_market", None)  # set only on factorized models
        dim_a = getattr(cfg, "latent_dim_asset",  None)

        # ── Non-factorized: original behaviour ────────────────────────────────
        if dim_m is None or dim_a is None:
            idx = np.random.choice(len(self._mu), size=n, replace=True)
            eps = np.random.randn(*self._mu[idx].shape).astype(np.float32)
            z_np = self._mu[idx] + eps * np.exp(0.5 * self._log_var[idx])
            return torch.tensor(z_np, dtype=torch.float32).to(self.device)

        # ── Factorized: split z_market (posterior) + z_asset (independent N(0,1)) ──
        T       = cfg.latent_length
        Q       = cfg.data_dim
        n_mkt   = dim_m * T          # length of z_market slice in full z vector
        n_asset = dim_a * Q          # length of z_asset slice in full z vector

        idx = np.random.choice(len(self._mu), size=n, replace=True)

        # z_market: sample from posterior
        mu_m   = self._mu[idx, :n_mkt]
        lv_m   = self._log_var[idx, :n_mkt]
        eps_m  = np.random.randn(n, n_mkt).astype(np.float32)
        z_mkt  = mu_m + eps_m * np.exp(0.5 * lv_m)

        # z_asset: fully independent standard normal — one per asset per scenario
        z_ast  = np.random.randn(n, n_asset).astype(np.float32)

        z_np = np.concatenate([z_mkt, z_ast], axis=1)   # (n, n_mkt + n_asset)
        return torch.tensor(z_np, dtype=torch.float32).to(self.device)

    def decoder_diversity_score(self, n: int = 200) -> float:
        """
        Diagnostic: mean per-asset std of decoded stage-1 log-returns across n
        posterior samples.  Values near zero indicate decoder collapse (the model
        ignores z and produces the same sequence regardless of input).
        Healthy models typically score >= 0.01 (1% log-return spread).
        """
        self.model.eval()
        with torch.no_grad():
            z_t = self._sample_z(n)
            idx = np.random.choice(len(self._all_conds), size=n, replace=True)
            c_t = torch.tensor(self._all_conds[idx], dtype=torch.float32).to(self.device)
            recon = self.model.decoder(z_t, c_t)["reconstruction"]
            log_ret1 = recon[:, 1, :].cpu().numpy()  # (n, n_assets)
        return float(log_ret1.std(axis=0).mean())

    def _vol_match(self, returns: np.ndarray) -> np.ndarray:
        """Standardise returns to historical week-1 mean and std."""
        gen_mean = returns.mean(axis=0)
        gen_std  = returns.std(axis=0).clip(1e-8)
        return (returns - gen_mean) / gen_std * self._hist_std + self._hist_mean

    def _vol_match2(self, returns: np.ndarray) -> np.ndarray:
        """Standardise returns to historical week-2 conditional mean and std."""
        gen_mean = returns.mean(axis=0)
        gen_std  = returns.std(axis=0).clip(1e-8)
        return (returns - gen_mean) / gen_std * self._hist_std2 + self._hist_mean2

    def _iman_conover(self, returns: np.ndarray) -> np.ndarray:
        """
        Iman-Conover rank reordering: impose the historical cross-asset correlation
        structure on `returns` while preserving each asset's exact marginal distribution.

        The vol-match step corrects marginals but not correlations — the LSTM decoder
        maps a single latent z to one market regime, so all assets co-move in every
        generated path regardless of their true historical correlation.  With near-unit
        pairwise correlations the optimizer cannot benefit from diversification and
        always concentrates in the single lowest-CVaR stock.

        Algorithm (Iman & Conover, 1982):
          1. Draw Z ~ N(0, C_hist) via Cholesky — shape (N, Q).
          2. For each asset k, rank-sort `returns[:, k]` to match the ordering of Z[:, k].
        The result has exactly the original per-column values but reordered so that
        cross-column rank correlations approximate C_hist.

        Reference: Iman, R.L. & Conover, W.J. (1982). A distribution-free approach
        to inducing rank correlation among input variables. Communications in
        Statistics — Simulation and Computation, 11(3), 311-334.
        """
        N, Q = returns.shape
        Z = np.random.randn(N, Q).astype(np.float32) @ self._hist_corr_chol.T  # (N, Q)
        result = np.empty_like(returns)
        for k in range(Q):
            target_ranks = np.argsort(np.argsort(Z[:, k]))   # rank order from Z
            sorted_col   = np.sort(returns[:, k])
            result[:, k] = sorted_col[target_ranks]
        return result

    def sample(self, n: int, context=None, stage: int = 1) -> np.ndarray:
        """
        Generate n one-step return scenarios.

        Args:
            n:     number of scenarios
            stage: 1 → VAE week-1 returns, vol-matched to historical distribution
                   2 → uniform U[-10%, +10%] (matches sample_joint methodology)

        Returns:
            np.ndarray of shape (n, n_assets)
        """
        if stage == 2:
            # Avoid vol-match amplification artifact: the LSTM generates smooth
            # consecutive steps so week-2 increments have tiny raw variance;
            # vol-match then amplifies by hist_std/gen_std (5-10x), producing
            # unrealistic single-week crashes of -40% to -70%.
            # U[-10%,+10%] matches the sample_joint() methodology and keeps
            # evaluate nodes as bounded rebalancing noise around each recourse node.
            return np.random.uniform(-0.10, 0.10, size=(n, self.n_assets))

        self.model.eval()
        with torch.no_grad():
            z_t = self._sample_z(n)
            idx = np.random.choice(len(self._all_conds), size=n, replace=True)
            c_t = torch.tensor(self._all_conds[idx], dtype=torch.float32).to(self.device)
            recon = self.model.decoder(z_t, c_t)["reconstruction"]  # (n, seq_len, n_assets)
            log_ret = recon[:, 1, :].clamp(-0.693, 0.693)

        returns = (torch.exp(log_ret) - 1.0).cpu().numpy()  # (n, n_assets)
        return self._vol_match(returns)

    def sample_joint(self, n: int) -> tuple:
        """
        Sample n scenario pairs for the two-stage problem.

        Both stages are decoded from the SAME latent z, preserving causal temporal
        structure — a crash-regime z produces correlated stress at both stages.

        Stage-1: VAE decoder week-1 log-returns, vol-matched to historical week-1 stats.
        Stage-2: VAE decoder week-2 incremental log-returns (recon[:,2,:]-recon[:,1,:]),
                 vol-matched to historical week-2 conditional return stats (P(2)/P(1)-1).

        Using separate week-2 vol-match stats (not the week-1 stats) ensures each asset
        gets its own correct stage-2 marginal distribution.  This is essential: if all
        assets share a symmetric uniform stage-2, CVaR is identical across stocks and the
        optimizer degenerates to picking whichever asset has the highest week-1 hist_mean.

        Returns
        -------
        ret1 : (n, n_assets) — VAE week-1 returns, vol-matched to (hist_mean, hist_std)
        ret2 : (n, n_assets) — VAE week-2 conditional returns, vol-matched to (hist_mean2, hist_std2)
        """
        self.model.eval()
        with torch.no_grad():
            z_t = self._sample_z(n)
            idx = np.random.choice(len(self._all_conds), size=n, replace=True)
            c_t = torch.tensor(self._all_conds[idx], dtype=torch.float32).to(self.device)
            recon = self.model.decoder(z_t, c_t)["reconstruction"]  # (n, T, n_assets)
            log_ret1 = recon[:, 1, :].clamp(-0.693, 0.693)
            # Incremental log-return at week 2: log(P(2)/P(1)) = log(P(2)/P(0)) - log(P(1)/P(0))
            log_ret2 = (recon[:, 2, :] - recon[:, 1, :]).clamp(-0.693, 0.693)

        ret1 = self._vol_match( (torch.exp(log_ret1) - 1.0).cpu().numpy())
        ret2 = self._vol_match2((torch.exp(log_ret2) - 1.0).cpu().numpy())
        return ret1, ret2

    def sample_joint_historical(self, n: int) -> tuple:
        """
        Fallback: bootstrap (stage-1, stage-2) pairs directly from historical
        windows without using the VAE decoder.

        For each draw, a random training window is chosen. Stage-1 is week 1
        and stage-2 is week 2 from the SAME window, preserving the empirical
        joint distribution and cross-asset correlations without model assumptions.

        Use this when decoder_diversity_score() < ~0.005 (decoder collapsed).

        Returns
        -------
        ret1 : (n, n_assets) — historical week-1 simple returns  (stage-1)
        ret2 : (n, n_assets) — historical week-2 simple returns  (stage-2)
        """
        idx = np.random.choice(len(self._hist_windows), size=n, replace=True)
        w    = self._hist_windows[idx]        # (n, T, n_assets), normalised P(t)/P(0)
        ret1 = w[:, 1, :] - 1.0              # P(1)/P(0) - 1
        ret2 = (w[:, 2, :] / w[:, 1, :]) - 1.0  # P(2)/P(1) - 1
        return ret1, ret2


# ─── GARCH + vine copula baseline (He & Zhang 2024) — full implementation ─────

class GARCHVineCopula:
    """
    Full He & Zhang (2024) implementation.

    Marginals
    ---------
    AR(1) mean filter + per-asset model selection over
    GARCH(1,1) / GJR-GARCH(1,1) / EGARCH(1,1) with skewed-Student-t errors.
    Best model chosen by AIC.

    Copula
    ------
    R-vine with full family set: Gaussian, Student, Clayton, Gumbel, Frank, Joe
    and all 90/180/270-degree rotations (180-deg rotated Gumbel captures the
    upper-tail equity dependence that dominates their Tree 1 — Table 3).

    Scenario reduction
    ------------------
    K-means on N = k * oversampling raw scenarios → k cluster centroids.
    Nodal probabilities = cluster-size / N (with optional moment-matching LP).

    Sequential multi-stage simulation
    -----------------------------------
    For stage-2 nodes, the GARCH conditional variance is updated using each
    stage-1 node's realized return before sampling, preserving temporal dynamics.

    Interface
    ---------
    .sample(n, prev_returns=None)              → (n, d) decimal returns
    .sample_with_probs(n, prev_returns=None)   → (centroids (n,d), probs (n,))
    .has_sequential_sampling = True
    """

    has_sequential_sampling = True

    def __init__(
        self,
        returns: np.ndarray,
        truncation_level: int = 3,
        model_selection: bool = True,
        oversampling: int = 25,
        moment_matching: bool = False,
        beta_tail: float = 0.45,
    ):
        """
        Args:
            returns:          (T, n_assets) historical return matrix, decimal.
            truncation_level: vine truncation level (3 balances accuracy and speed).
            model_selection:  if True, select GARCH variant per asset via AIC.
            oversampling:     raw scenarios = k * oversampling before K-means.
            moment_matching:  if True, LP to adjust probs to match historical moments.
            beta_tail:        tail quantile for moment-matching (paper: 0.45).
        """
        try:
            from arch import arch_model as _check  # noqa
        except ImportError as e:
            raise ImportError("pip install arch") from e
        try:
            import pyvinecopulib as _check  # noqa
        except ImportError as e:
            raise ImportError("pip install pyvinecopulib") from e

        self.n_assets = returns.shape[1]
        self.truncation_level = truncation_level
        self.oversampling = max(1, oversampling)
        self.moment_matching = moment_matching
        self.beta_tail = beta_tail
        self._historical_returns = returns.copy()

        print(
            f"[GARCHVineCopula] Fitting {self.n_assets} marginals "
            f"(model_selection={model_selection})…"
        )
        self._marginal_info = []
        raw_resids = []

        for i in range(self.n_assets):
            info = self._fit_marginal(returns[:, i], i, model_selection)
            self._marginal_info.append(info)
            raw_resids.append(info["std_resid"])

        min_len = min(len(r) for r in raw_resids)
        resid_matrix = np.column_stack([r[-min_len:] for r in raw_resids])
        self._sorted_resids = np.sort(resid_matrix, axis=0)

        print("[GARCHVineCopula] Fitting R-vine copula…")
        pseudo_obs = self._empirical_pit(resid_matrix)
        family_set = self._build_family_set()
        import pyvinecopulib as pv
        controls = pv.FitControlsVinecop(
            family_set=family_set,
            trunc_lvl=truncation_level,
        )
        self.copula = pv.Vinecop(self.n_assets)
        self.copula.select(data=pseudo_obs, controls=controls)
        print("[GARCHVineCopula] Fitting complete.")

    # ── Family set ────────────────────────────────────────────────────────────

    @staticmethod
    def _build_family_set():
        import pyvinecopulib as pv

        bases = [
            "gaussian", "student",
            "clayton", "gumbel", "frank", "joe",
            "bb1", "bb6", "bb7", "bb8",
        ]
        rotations = ["", "_90", "_180", "_270"]
        families = []
        for b in bases:
            for r in rotations:
                fam = getattr(pv.BicopFamily, b + r, None)
                if fam is not None:
                    families.append(fam)
        return families if families else [pv.BicopFamily.gaussian, pv.BicopFamily.student]

    # ── Marginal fitting ──────────────────────────────────────────────────────

    def _fit_marginal(self, returns_1d: np.ndarray, idx: int, model_selection: bool) -> dict:
        from arch import arch_model

        r_pct = returns_1d * 100.0  # % scale improves GARCH numerical stability

        # (name, vol kwarg, asymmetric order)
        candidates = [
            ("GJR-GARCH", "GARCH",  1),
            ("GARCH",     "GARCH",  0),
            ("EGARCH",    "EGARCH", 1),
        ] if model_selection else [("GJR-GARCH", "GARCH", 1)]

        best_aic = np.inf
        best_res = None
        best_name = "GJR-GARCH"

        for name, vol, o in candidates:
            try:
                am = arch_model(
                    r_pct, mean="AR", lags=1,
                    vol=vol, p=1, o=o, q=1,
                    dist="skewt", rescale=False,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = am.fit(disp="off")
                if np.isfinite(res.aic) and res.aic < best_aic:
                    best_aic = res.aic
                    best_res = res
                    best_name = name
            except Exception:
                continue

        if best_res is None:
            am = arch_model(r_pct, vol="GARCH", p=1, q=1, dist="Normal", rescale=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best_res = am.fit(disp="off")
            best_name = "GARCH"

        params = best_res.params

        def _p(key, default=0.0):
            for k in (key, key.lower()):
                if k in params.index:
                    return float(params[k])
            return default

        # AR(1) mean parameters
        ar_const = _p("Const", _p("mu", _p("constant", 0.0)))
        ar_phi   = _p("r[1]",  _p("phi[1]", _p("Lag1", 0.0)))

        # GARCH parameters
        omega = _p("omega",   1e-4)
        alpha = _p("alpha[1]", 0.05)
        beta  = _p("beta[1]",  0.90)
        gamma = _p("gamma[1]", 0.0)  # asymmetry / leverage

        # Enforce GARCH/GJR-GARCH stationarity: alpha + beta must be < 1.
        # EGARCH stationarity only requires |beta| < 1, so skip rescaling there.
        if best_name in ("GARCH", "GJR-GARCH"):
            persistence = alpha + beta
            if persistence >= 1.0:
                scale = 0.99 / persistence
                alpha *= scale
                beta  *= scale

        # Standardised residuals
        std_r = best_res.std_resid
        if hasattr(std_r, "values"):
            std_r = std_r.values
        std_r = std_r.astype(float)
        std_r = std_r[np.isfinite(std_r)]

        # One-step-ahead conditional variance h_{T+1}
        try:
            fc = best_res.forecast(horizon=1)
            last_h = float(fc.variance.values[-1, 0])
        except Exception:
            cv = best_res.conditional_volatility
            sigma_T = float(cv.iloc[-1] if hasattr(cv, "iloc") else cv[-1])
            last_h = sigma_T ** 2

        # Cap last_h at 4× the unconditional variance to prevent transient
        # volatility spikes from producing implausibly extreme scenarios.
        unconditional_var = float(np.var(r_pct))
        last_h = min(last_h, 4.0 * unconditional_var)
        last_h = max(last_h, 1e-8)

        return {
            "vol":       best_name,
            "aic":       best_aic,
            "std_resid": std_r,
            "ar_const":  ar_const,
            "ar_phi":    ar_phi,
            "omega":     omega,
            "alpha":     alpha,
            "beta":      beta,
            "gamma":     gamma,
            "last_h":    last_h,           # h_{T+1}: used for stage-1 cond vol
            "last_z":    float(std_r[-1]) if len(std_r) > 0 else 0.0,
            "last_r_pct": float(r_pct[-1]),
        }

    # ── Helper: empirical PIT and inverse ─────────────────────────────────────

    @staticmethod
    def _empirical_pit(data: np.ndarray) -> np.ndarray:
        n = data.shape[0]
        ranks = np.apply_along_axis(
            lambda col: np.argsort(np.argsort(col)) + 1, axis=0, arr=data
        )
        return np.clip(ranks / (n + 1), 1e-7, 1 - 1e-7)

    def _invert_pit(self, u_sim: np.ndarray) -> np.ndarray:
        n_fit = self._sorted_resids.shape[0]
        q_lev = np.arange(1, n_fit + 1) / (n_fit + 1)
        out = np.empty_like(u_sim)
        for i in range(self.n_assets):
            out[:, i] = np.interp(u_sim[:, i], q_lev, self._sorted_resids[:, i])
        return out

    # ── GARCH one-step state update ───────────────────────────────────────────

    def _update_h(self, info: dict, prev_return_pct: float) -> float:
        """
        Compute h_{T+2} given the stage-1 return (in %).

        info["last_h"] = h_{T+1}  (stored as one-step forecast from fitting)
        prev_return_pct           = r_1 (stage-1 return, %)
        """
        h_t   = info["last_h"]
        omega = info["omega"]
        alpha = info["alpha"]
        beta  = info["beta"]
        gamma = info["gamma"]
        vol   = info["vol"]

        mu_t  = info["ar_const"] + info["ar_phi"] * info["last_r_pct"]
        eps_t = prev_return_pct - mu_t

        if vol == "EGARCH":
            z_t = eps_t / (np.sqrt(h_t) + 1e-12)
            # Use empirical E[|z|] from the fitted residuals (correct for skewed-t;
            # sqrt(2/pi) = 0.7979 is only exact for standard normal).
            E_abs_z = float(np.mean(np.abs(info.get("std_resid", np.array([0.7979432])))))
            if not np.isfinite(E_abs_z) or E_abs_z <= 0:
                E_abs_z = 0.7979432
            log_h = (omega
                     + alpha * (abs(z_t) - E_abs_z)
                     + gamma * z_t
                     + beta * np.log(max(h_t, 1e-12)))
            h_new = np.exp(log_h)
        else:  # GARCH or GJR-GARCH
            ind = 1.0 if eps_t < 0 else 0.0
            h_new = omega + (alpha + gamma * ind) * eps_t ** 2 + beta * h_t

        return max(float(h_new), 1e-8)

    # ── Core raw sampler ──────────────────────────────────────────────────────

    def _raw_sample(self, n: int, prev_returns_pct=None) -> np.ndarray:
        """
        Generate n decimal return scenarios.

        prev_returns_pct: (n_assets,) stage-1 returns in %, or None for stage-1.
        """
        cond_vols = np.empty(self.n_assets)
        ar_means  = np.empty(self.n_assets)

        for i, info in enumerate(self._marginal_info):
            if prev_returns_pct is not None:
                h2 = self._update_h(info, float(prev_returns_pct[i]))
                cond_vols[i] = np.sqrt(h2)
                ar_means[i]  = info["ar_const"] + info["ar_phi"] * prev_returns_pct[i]
            else:
                cond_vols[i] = np.sqrt(info["last_h"])
                ar_means[i]  = info["ar_const"] + info["ar_phi"] * info["last_r_pct"]

        u_sim   = self.copula.simulate(n)              # (n, d) uniform marginals
        std_eps = self._invert_pit(u_sim)              # (n, d) standardised residuals
        ret_pct = std_eps * cond_vols + ar_means       # (n, d) returns in %
        return ret_pct / 100.0                         # decimal

    # ── Public interface ──────────────────────────────────────────────────────

    def sample(self, n: int, prev_returns=None, context=None) -> np.ndarray:
        """
        Generate n one-step return scenarios (decimal, shape (n, n_assets)).

        prev_returns: (n_assets,) stage-1 decimal returns for GARCH state update.
        """
        prev_pct = prev_returns * 100.0 if prev_returns is not None else None
        return self._raw_sample(n, prev_returns_pct=prev_pct)

    def sample_with_probs(
        self, n: int, prev_returns=None
    ):
        """
        Generate n scenario nodes using K-means reduction.

        Returns
        -------
        centroids : (n, n_assets)  decimal return scenarios
        probs     : (n,)           nodal probabilities summing to 1
        """
        from sklearn.cluster import KMeans

        N = n * self.oversampling
        prev_pct = prev_returns * 100.0 if prev_returns is not None else None
        raw = self._raw_sample(N, prev_returns_pct=prev_pct)   # (N, d)

        if n >= N or n == 1:
            probs = np.full(min(n, len(raw)), 1.0 / min(n, len(raw)))
            return raw[:n], probs

        km = KMeans(n_clusters=n, n_init=10, max_iter=300, random_state=42)
        labels = km.fit_predict(raw)
        centroids = km.cluster_centers_

        counts = np.bincount(labels, minlength=n).astype(float)
        probs = counts / counts.sum()

        if self.moment_matching:
            probs = self._moment_match(centroids, probs)

        return centroids, probs

    # ── Optional moment-matching LP ───────────────────────────────────────────

    def _moment_match(self, centroids: np.ndarray, init_probs: np.ndarray) -> np.ndarray:
        """
        LP: min Σ|p_i - p0_i|  s.t. per-asset mean and E[R²] match historical
        values within 10% tolerance.  Falls back to init_probs on failure.
        """
        import cvxpy as cp

        K = len(init_probs)
        hist = self._historical_returns
        hist_mean = hist.mean(axis=0)
        e_r2 = hist.var(axis=0) + hist_mean ** 2

        p = cp.Variable(K, nonneg=True)
        d = cp.Variable(K, nonneg=True)

        tol = 0.10
        constraints = [cp.sum(p) == 1, d >= p - init_probs, d >= init_probs - p]

        for j in range(self.n_assets):
            c = centroids[:, j]
            mu  = float(hist_mean[j])
            slack = max(abs(mu) * tol, 1e-5)
            constraints += [p @ c >= mu - slack, p @ c <= mu + slack]

            er2   = float(e_r2[j])
            constraints += [
                p @ (c ** 2) >= max(er2 * (1 - tol), 0),
                p @ (c ** 2) <= er2 * (1 + tol) + 1e-8,
            ]

        prob = cp.Problem(cp.Minimize(cp.sum(d)), constraints)
        try:
            prob.solve(solver=cp.HIGHS, verbose=False)
        except Exception:
            return init_probs

        if prob.status in ("optimal", "optimal_inaccurate") and p.value is not None:
            result = np.maximum(p.value, 0.0)
            s = result.sum()
            if s > 0:
                return result / s

        return init_probs


# ─── TC-VAE + Vine Copula hybrid ──────────────────────────────────────────────

class TCVAEVineCopula:
    """
    Hybrid generator: TC-VAE marginals + vine copula dependence structure.

    Motivation
    ----------
    TC-VAE generates realistic per-asset marginal distributions (fat tails,
    volatility clustering, temporal dynamics) but over-estimates cross-asset
    correlation (~0.78 vs historical ~0.35) because a single latent z drives
    all 30 assets.

    The vine copula (fit to historical returns) provides the correct dependence
    structure.  Combining them via the Sklar decomposition gives scenarios with:
      - Correct cross-asset dependence  (from vine copula)
      - Correct per-asset dynamics      (from TC-VAE marginals)

    Algorithm
    ---------
    Initialisation:
      1. Pre-generate a large pool (N_pool) of (ret1, ret2) pairs from the
         TC-VAE — these define the empirical marginal CDF F̂_k for each asset k.
      2. Fit an R-vine copula C to historical week-1 returns via the empirical
         probability integral transform (PIT).  This captures the historical
         cross-asset rank dependence without parametric marginal assumptions.

    sample_joint(n):
      Stage-1
        a. Sample n uniform vectors (u_1,...,u_Q) ~ C  (vine copula).
        b. Apply inverse TC-VAE empirical CDF per asset:
               r1_k = F̂_k^{-1}(u_k)   ← quantile interpolation into pool
           Result: correct vine dependence + TC-VAE marginals.

      Stage-2 (conditional on stage-1 centroid)
        c. For each of the n stage-1 draws, find its k nearest neighbours
           in the TC-VAE pool by 1-D market return (cross-sectional mean).
           This subset is the conditional TC-VAE pool for that draw.
        d. Sample vine copula uniforms fitted to historical stage-2 returns.
        e. Apply inverse empirical CDF using the conditional pool.
           Result: stage-2 inherits TC-VAE temporal dynamics (stage-1 regime
           influences stage-2 via k-NN conditioning) + vine dependence.

    Note on fitting the vine in latent (z) space
    ---------------------------------------------
    Rejected: z_asset components are trained toward N(0,I) so a vine there
    approximates a Gaussian copula and adds no information.  More importantly
    we want dependence in *return* space, not in the 232-dim latent space.
    Historical returns are the natural and correct fitting target.
    """

    has_sequential_sampling = False

    def __init__(
        self,
        tcvae_generator: "DirectTCVAE",
        historical_returns: np.ndarray,
        pool_size: int = 5000,
        n_neighbours: int = 500,
        truncation_level: int = 5,
        seed: int = 42,
    ):
        """
        Args:
            tcvae_generator:    fitted DirectTCVAE instance
            historical_returns: (T, Q) decimal weekly returns for vine fitting
            pool_size:          TC-VAE sample pool size (larger = better empirical CDF)
            n_neighbours:       k-NN pool size for stage-2 conditioning
            truncation_level:   vine truncation (5 fits trees 1-5; captures ~90% of
                                pairwise dependence while avoiding overfitting on 496
                                samples; level 3 undershoots historical correlation
                                by ~30% and causes negative in-sample CVaR)
            seed:               RNG seed used only for vine fitting, not simulation
        """
        try:
            import pyvinecopulib as pv
        except ImportError:
            raise ImportError("pip install pyvinecopulib")

        self.tcvae = tcvae_generator
        self.n_assets = tcvae_generator.n_assets
        self.n_neighbours = min(n_neighbours, pool_size - 1)
        self._seed = seed
        Q = self.n_assets

        # ── 1. Pre-generate TC-VAE pool ────────────────────────────────────────
        print(f"[TCVAEVineCopula] Generating TC-VAE pool (n={pool_size})…")
        self._ret1_pool, self._ret2_pool = tcvae_generator.sample_joint(pool_size)
        # 1-D market projection for k-NN conditioning (cross-sectional mean)
        self._mkt1_pool = self._ret1_pool.mean(axis=1)   # (pool_size,)

        # ── 2. Fit vine copula to historical stage-1 returns ──────────────────
        T = historical_returns.shape[0]
        print(f"[TCVAEVineCopula] Fitting R-vine on {T} historical returns "
              f"({Q} assets, trunc={truncation_level})…")

        u1 = self._empirical_pit(historical_returns)
        controls = pv.FitControlsVinecop(
            family_set=pv.all,
            trunc_lvl=truncation_level,
            selection_criterion="bic",
        )
        self._vine1 = pv.Vinecop(Q)
        self._vine1.select(data=u1, controls=controls)

        # Also fit vine to stage-2 returns (week-2 conditional returns)
        # Use pool stage-2 as a proxy (captures TC-VAE stage-2 marginal structure)
        # For the vine we use historical data to get the right rank dependence
        hist_ret2 = historical_returns  # same historical data — vine captures rank structure
        u2 = self._empirical_pit(hist_ret2)
        self._vine2 = pv.Vinecop(Q)
        self._vine2.select(data=u2, controls=controls)

        print("[TCVAEVineCopula] Ready.")

    @staticmethod
    def _empirical_pit(returns: np.ndarray) -> np.ndarray:
        """Hazen empirical probability integral transform → (T, Q) uniforms in (0,1)."""
        T, Q = returns.shape
        u = np.empty_like(returns)
        for k in range(Q):
            ranks = np.argsort(np.argsort(returns[:, k])) + 1   # 1-indexed ranks
            u[:, k] = ranks / (T + 1)
        return u

    @staticmethod
    def _apply_empirical_quantile(u: np.ndarray, pool: np.ndarray) -> np.ndarray:
        """
        Map (n, Q) vine uniform samples to (n, Q) returns via empirical quantile
        of pool using linear interpolation (Hazen grid).

        This is the key Sklar step: replaces parametric GARCH marginals with
        the TC-VAE empirical marginal distribution.
        """
        n, Q = u.shape
        N_pool = pool.shape[0]
        grid = (np.arange(1, N_pool + 1) - 0.5) / N_pool
        result = np.empty((n, Q), dtype=np.float32)
        for k in range(Q):
            result[:, k] = np.interp(u[:, k], grid, np.sort(pool[:, k]))
        return result

    def sample_joint(self, n: int) -> tuple:
        """
        Generate n (ret1, ret2) scenario pairs.

        Returns
        -------
        ret1 : (n, Q) — stage-1 decimal returns with vine dependence + TC-VAE marginals
        ret2 : (n, Q) — stage-2 decimal returns, stage-1 conditioned via k-NN
        """
        import pyvinecopulib as pv

        # ── Stage-1 ────────────────────────────────────────────────────────────
        # No fixed seed: each call draws fresh vine uniforms so that different
        # tree builds sample genuinely different realisations of the dependence
        # structure.  A fixed seed caused all trees to share the same 15k-scenario
        # pool, letting the optimizer overfit to that single vine realization
        # (cross-method CVaR 3-4x higher than in-sample).
        u1 = self._vine1.simulate(n=n)                               # (n, Q) uniform
        ret1 = self._apply_empirical_quantile(u1, self._ret1_pool)   # (n, Q)

        # ── Stage-2 (k-NN conditional on stage-1 market return) ───────────────
        mkt1 = ret1.mean(axis=1)   # (n,) each scenario's market return at stage-1
        ret2 = np.empty_like(ret1)

        u2 = self._vine2.simulate(n=n)                               # (n, Q) uniform

        for i in range(n):
            # Find k nearest TC-VAE pool entries by stage-1 market return
            dists = np.abs(self._mkt1_pool - mkt1[i])
            nn_idx = np.argpartition(dists, self.n_neighbours)[:self.n_neighbours]

            cond_pool = self._ret2_pool[nn_idx]                       # (k, Q) pool

            # Apply inverse empirical CDF of the conditional pool
            N_k = len(nn_idx)
            grid = (np.arange(1, N_k + 1) - 0.5) / N_k
            for k in range(self.n_assets):
                ret2[i, k] = np.interp(u2[i, k], grid, np.sort(cond_pool[:, k]))

        return ret1, ret2


# ─── Registry ──────────────────────────────────────────────────────────────────

COPULAS = {
    "gaussian": GaussianCopula,
    "student_t": StudentTCopula,
    "clayton": ClaytonCopula,
}
