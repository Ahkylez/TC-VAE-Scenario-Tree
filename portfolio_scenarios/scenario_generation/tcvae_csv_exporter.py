"""
Export a causally-consistent TC-VAE two-stage scenario tree to CSV.

The C++ solver expects one row per evaluate node, no header:

    col 0           : recourse node index j (0-indexed)
    col 1           : p_j   — recourse node probability
    col 2           : p_je  — evaluate node conditional probability
    cols 3..Q+2     : P^j_i — recourse node prices (Q assets)
    cols Q+3..2Q+2  : P^(j,e)_i — evaluate node prices (Q assets)

Causal construction
-------------------
The existing ScenarioTree.build(is_tcvae=True) draws a fresh z for every
call to generator.sample(), so stage-2 scenarios are independent of the
z that produced stage-1 — causality is broken.

The correct approach used here:

  1. Draw N = J * oversampling * E paths from the prior in one batch.
     Each path i has a (stage1_return_i, stage2_return_i) pair decoded
     from the SAME latent z_i.  A crash-regime z produces correlated
     stress at both stages.

  2. K-medoids cluster the N stage-1 return vectors into J clusters.
     Each cluster's medoid IS an actual generated scenario (not a
     centroid average), so per-asset return variance is fully preserved.

  3. Within cluster j, the stage-2 returns of its members are the
     evaluate children of node j.  If a cluster has fewer than E
     members, resample its stage-2 rows with replacement to fill E slots.

  4. p_j  = 1 / J  (uniform — K-medoids positioning already encodes
                   distributional structure; density-weighting would
                   double-count common regimes)
     p_je = 1 / E  (equal weight within cluster)

Usage
-----
    from portfolio_scenarios.scenario_generation.tcvae_csv_exporter import build_tcvae_tree_csv
    from experiments.experiment_utils import load_trained_model
    from portfolio_scenarios.data_pipeline.sp100_dataset import SP100Dataset

    dataset = SP100Dataset(...)
    generator = load_trained_model(model_dir, dataset)   # returns DirectTCVAE

    build_tcvae_tree_csv(
        generator      = generator,
        initial_prices = dataset.initial_prices,
        num_recourse   = 400,
        num_evaluate   = 40,
        output_path    = "data/scenarios/tcvae_tree.csv",
    )
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kmedoids_on_subset(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """
    K-means on X then snap each centroid to the nearest real point in X.
    Returns medoid_indices (length k) — indices into X.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import euclidean_distances

    km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=seed)
    km.fit(X)
    centroids = km.cluster_centers_   # (k, Q)

    chunk = 4096
    best_dist  = np.full(k, np.inf)
    medoid_idx = np.zeros(k, dtype=int)
    for start in range(0, len(X), chunk):
        end = min(start + chunk, len(X))
        d = euclidean_distances(centroids, X[start:end])
        local_best  = d.argmin(axis=1)
        local_dists = d[np.arange(k), local_best]
        improved = local_dists < best_dist
        best_dist[improved]  = local_dists[improved]
        medoid_idx[improved] = start + local_best[improved]

    return medoid_idx


def _kmedoids_with_voronoi_probs(
    X: np.ndarray,
    k: int,
    seed: int = 42,
) -> tuple:
    """
    K-medoids on the full scenario set X with Voronoi-cell proportional
    probabilities.

    Runs K-means to find k cluster centres, snaps each centre to the nearest
    real data point (making them true medoids), assigns every scenario to its
    nearest medoid, then sets p_j = (cluster_size_j / N).

    Why Voronoi probabilities instead of uniform 1/J
    ------------------------------------------------
    Uniform 1/J systematically overweights sparse body medoids and
    underweights tail medoids.  For low-mean stocks (true weekly mean ≈ 0.1%)
    this produces 5-10× mean inflation: the body scenarios dominate the
    weighted average while the (correctly negative) tail scenarios only
    contribute 10% weight despite representing 15% of the distribution.

    With Voronoi-cell probabilities p_j ∝ cluster size the weighted mean of
    the J medoids equals the sample mean of the full pool (exact for K-means
    centres, approximate for K-medoids because medoids are snapped to real
    points).  This is exactly the property that Wasserstein backward reduction
    (Dupacova et al. 2003) preserves through each removal step.

    Tail coverage: K-medoids naturally places medoids proportional to density.
    For 95% CVaR with J=100, roughly 5 medoids fall in the worst-5% tail,
    each with natural probability ~1% — adequate for CVaR estimation.

    Parameters
    ----------
    X    : (N, Q) stage-1 return matrix
    k    : number of recourse nodes J
    seed : RNG seed

    Returns
    -------
    labels     : (N,) cluster index for each scenario
    medoid_idx : (k,) indices into X (the k representative scenarios)
    probs      : (k,) Voronoi-cell probabilities (sum to 1.0)
    """
    from sklearn.metrics.pairwise import euclidean_distances

    n = len(X)
    medoid_idx = _kmedoids_on_subset(X, k, seed=seed)

    # Assign every scenario to its nearest medoid
    medoid_scenarios = X[medoid_idx]               # (k, Q)
    labels = np.empty(n, dtype=int)
    for start in range(0, n, 4096):
        end = min(start + 4096, n)
        d = euclidean_distances(medoid_scenarios, X[start:end])  # (k, chunk)
        labels[start:end] = d.argmin(axis=0)

    counts = np.bincount(labels, minlength=k).astype(np.float64)
    probs  = counts / counts.sum()

    print(f"[tcvae_csv]   K-medoids: {k} medoids, "
          f"prob range [{probs.min():.4f}, {probs.max():.4f}]  "
          f"(min cluster {int(counts.min())}, max {int(counts.max())} scenarios)")

    return labels, medoid_idx, probs


# ── Main export function ───────────────────────────────────────────────────────

def build_tcvae_tree_csv(
    generator,
    initial_prices: np.ndarray,
    num_recourse: int,
    num_evaluate: int,
    output_path: str,
    oversampling: int = 5,
    seed: int | None = None,
    use_vae: bool | None = None,
    collapse_threshold: float = 0.005,
    use_reduction: bool = True,
) -> None:
    """
    Build a causally-consistent two-stage scenario tree and export it to CSV
    for the C++ CVaR solver.

    Parameters
    ----------
    generator          : DirectTCVAE instance
    initial_prices     : (Q,) array of t=0 prices for all Q assets
    num_recourse       : number of stage-1 (recourse) nodes J
    num_evaluate       : number of stage-2 (evaluate) children per recourse node E
    output_path        : destination CSV path (directories created if missing)
    oversampling       : total samples drawn = J * oversampling * E
                         Ignored when use_reduction=False (pool = J * E exactly).
    seed               : numpy RNG seed for reproducibility
    use_vae            : True  → force VAE decoder sampling
                         False → force historical bootstrap
                         None  → auto-detect via decoder_diversity_score()
    collapse_threshold : diversity score below this triggers fallback to
                         historical bootstrap (default 0.005)
    use_reduction      : True  (default) → K-medoids on J*oversampling*E pool with
                                           Voronoi-cell proportional probabilities.
                         False → direct sort-based assignment on J*E scenarios with
                                 uniform p_j = 1/J.  Faster; appropriate when J*E
                                 is already large (≥ 10,000) and oversampling adds
                                 cost without meaningful coverage improvement.
    """
    if seed is not None:
        np.random.seed(seed)

    Q = len(initial_prices)

    # ── Auto-detect decoder collapse ─────────────────────────────────────────
    # TCVAEVineCopula and other hybrid generators expose only sample_joint();
    # skip the diversity check and historical fallback for those.
    has_diversity_score   = hasattr(generator, "decoder_diversity_score")
    has_historical_sample = hasattr(generator, "sample_joint_historical")

    if use_vae is None:
        if has_diversity_score:
            score = generator.decoder_diversity_score(n=500)
            print(f"[tcvae_csv] Decoder diversity score: {score:.5f} "
                  f"(threshold={collapse_threshold})")
            if score < collapse_threshold:
                print(f"[tcvae_csv] WARNING: Decoder appears collapsed (score < {collapse_threshold}).")
                print(f"[tcvae_csv] Falling back to historical bootstrap. Retrain with a larger")
                print(f"[tcvae_csv] beta (try 0.1-0.5) to fix this.")
                use_vae = False
            else:
                print(f"[tcvae_csv] Decoder diversity OK -- using VAE sampling.")
                use_vae = True
        else:
            print(f"[tcvae_csv] Generator has no diversity score (hybrid/copula) -- "
                  f"using sample_joint() directly.")
            use_vae = True

    _os_str = f"oversampling={oversampling}" if use_reduction else "no reduction"
    # use_reduction=True : oversample J*oversampling*E paths, then K-medoids compress to J nodes
    # use_reduction=False: sample exactly J*E pairs, direct sort assignment
    N = num_recourse * num_evaluate * oversampling if use_reduction else num_recourse * num_evaluate
    print(f"[tcvae_csv] Sampling {N} joint paths (J={num_recourse}, E={num_evaluate}, "
          f"{_os_str}, use_vae={use_vae})...")

    if use_vae:
        ret1, ret2 = generator.sample_joint(N)
    else:
        if not has_historical_sample:
            raise ValueError(
                "Generator diversity score is below threshold but the generator "
                "has no sample_joint_historical() fallback. Retrain or pass use_vae=True."
            )
        ret1, ret2 = generator.sample_joint_historical(N)

    corr1 = np.corrcoef(ret1.T)
    mask  = ~np.eye(corr1.shape[0], dtype=bool)
    p5_1  = np.percentile(ret1, 5,  axis=0)
    p95_1 = np.percentile(ret1, 95, axis=0)
    mkt1  = ret1.mean(axis=1)   # cross-sectional market return per scenario
    print(f"[tcvae_csv] Stage-1 pool ({N} scenarios):")
    print(f"  per-asset  mean : min={ret1.mean(0).min():.4f}  max={ret1.mean(0).max():.4f}")
    print(f"  per-asset  std  : min={ret1.std(0).min():.4f}  max={ret1.std(0).max():.4f}")
    print(f"  per-asset  5pct : min={p5_1.min():.4f}  max={p5_1.max():.4f}")
    print(f"  per-asset  95pct: min={p95_1.min():.4f}  max={p95_1.max():.4f}")
    print(f"  mkt return 5pct={np.percentile(mkt1,5):.4f}  50pct={np.percentile(mkt1,50):.4f}  95pct={np.percentile(mkt1,95):.4f}")
    print(f"  pairwise corr: min={corr1[mask].min():.3f}  mean={corr1[mask].mean():.3f}  max={corr1[mask].max():.3f}")
    print(f"[tcvae_csv] Stage-2 pool:")
    print(f"  per-asset  mean : min={ret2.mean(0).min():.4f}  max={ret2.mean(0).max():.4f}")
    print(f"  per-asset  std  : min={ret2.std(0).min():.4f}  max={ret2.std(0).max():.4f}")

    # ── Assemble CSV rows ─────────────────────────────────────────────────────
    n_rows = num_recourse * num_evaluate
    n_cols = 3 + 2 * Q
    out = np.empty((n_rows, n_cols), dtype=np.float64)

    p_je = 1.0 / num_evaluate
    row  = 0
    rng  = np.random.default_rng(seed)

    if use_vae and use_reduction:
        # K-medoids on the full oversampled pool with Voronoi-cell proportional
        # probabilities.  Each medoid IS a real generated scenario (not a centroid
        # average), so per-asset return variance is fully preserved.  p_j ∝ cluster
        # size so the weighted mean of the J medoids matches the pool mean exactly.
        labels, medoid_idx, p_j = _kmedoids_with_voronoi_probs(ret1, num_recourse,
                                                                seed=seed or 42)

        for j in range(num_recourse):
            recourse_prices = initial_prices * (1.0 + ret1[medoid_idx[j]])

            member_idx = np.where(labels == j)[0]
            chosen = rng.choice(
                member_idx, size=num_evaluate,
                replace=(len(member_idx) < num_evaluate),
            ) if len(member_idx) > 0 else rng.choice(N, size=num_evaluate, replace=True)

            for e_idx in chosen:
                eval_prices = recourse_prices * (1.0 + ret2[e_idx])
                out[row, 0] = j
                out[row, 1] = p_j[j]
                out[row, 2] = p_je
                out[row, 3          : 3 + Q]     = recourse_prices
                out[row, 3 + Q      : 3 + 2 * Q] = eval_prices
                row += 1

    elif use_vae and not use_reduction:
        # Direct sort-based assignment — no clustering.
        #
        # Sort all N = J*E scenarios by stage-1 cross-sectional market return so
        # that consecutive groups of E share a similar regime.  The middle scenario
        # of each group becomes the recourse representative; its E group-mates
        # supply the evaluate children.  Because each (ret1_i, ret2_i) is a causally
        # consistent pair from the same vine draw, stage-2 conditioning is preserved
        # within each group (all members have similar stage-1 market returns, so
        # their k-NN pools overlap substantially).
        #
        # p_j = 1/J (uniform) — with J*E scenarios the vine's own sampling
        # distribution provides adequate moment coverage without Voronoi weighting.
        print(f"[tcvae_csv] Direct sort-based assignment (no reduction), "
              f"J={num_recourse}, E={num_evaluate}...")
        mkt1  = ret1.mean(axis=1)                     # (N,) cross-sectional mean
        order = np.argsort(mkt1)                      # sort by market return
        p_j   = np.full(num_recourse, 1.0 / num_recourse)

        for j in range(num_recourse):
            group = order[j * num_evaluate : (j + 1) * num_evaluate]
            mid   = group[len(group) // 2]            # median-return scenario as recourse
            recourse_prices = initial_prices * (1.0 + ret1[mid])

            for e_idx in group:
                eval_prices = recourse_prices * (1.0 + ret2[e_idx])
                out[row, 0] = j
                out[row, 1] = p_j[j]
                out[row, 2] = p_je
                out[row, 3          : 3 + Q]     = recourse_prices
                out[row, 3 + Q      : 3 + 2 * Q] = eval_prices
                row += 1

    else:
        # Historical bootstrap path: direct sampling — NO K-medoids compression.
        #
        # K-means/K-medoids centroids of ~(N/J) samples each would collapse
        # per-asset weekly variance from ~4% to ~0.3%.  All 400 nodes would land
        # near the historical mean, so the CVaR optimizer cannot differentiate
        # scenarios and always selects the minimum-volatility asset.
        #
        # Instead, randomly select J recourse nodes from the N pre-sampled stage-1
        # returns directly, preserving the full historical return distribution.
        # Stage-2 evaluate nodes are drawn independently (different window indices),
        # matching the independence assumption used by the copula scenario tree.
        p_j = np.full(num_recourse, 1.0 / num_recourse)
        rec_idx = rng.choice(N, size=num_recourse, replace=False)

        for j in range(num_recourse):
            recourse_prices = initial_prices * (1.0 + ret1[rec_idx[j]])
            eval_idx = rng.choice(N, size=num_evaluate, replace=True)
            for e in range(num_evaluate):
                eval_prices = recourse_prices * (1.0 + ret2[eval_idx[e]])
                out[row, 0] = j
                out[row, 1] = p_j[j]
                out[row, 2] = p_je
                out[row, 3          : 3 + Q]     = recourse_prices
                out[row, 3 + Q      : 3 + 2 * Q] = eval_prices
                row += 1

    # ── Write CSV ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savetxt(output_path, out, delimiter=",", fmt="%.8f")

    print(
        f"[tcvae_csv] Exported {n_rows} rows to '{output_path}'\n"
        f"  {num_recourse} recourse nodes x {num_evaluate} evaluate nodes, "
        f"{Q} assets\n"
        f"  p_j range : [{p_j.min():.5f}, {p_j.max():.5f}]  "
        f"(sum={p_j.sum():.6f})\n"
        f"  Row layout: [j | p_j | p_je | {Q}x P^j | {Q}x P^(j,e)]"
    )


# ── Optimal node count finder ─────────────────────────────────────────────────

def find_optimal_tree_size(
    generator,
    J_candidates: list | None = None,
    n_pool: int = 10_000,
    seed: int = 42,
    plot: bool = True,
    save_path: str | None = None,
) -> dict:
    """
    Find the optimal number of recourse nodes J via the Wasserstein elbow method.

    Samples n_pool stage-1 returns from the generator, then for each J in
    J_candidates runs K-medoids clustering and records the mean quantization
    error (mean L2 distance from each scenario to its nearest medoid).  This
    is a proxy for the Wasserstein-1 distance between the full distribution
    and the J-node approximation — it decreases monotonically and exhibits an
    elbow at the point of diminishing returns.

    Evaluate nodes E come from the week-2 VAE decoder output, which carries
    real distributional structure correlated with stage-1.  The elbow method
    here applies only to J (recourse nodes); for E, 10-20 is typically
    sufficient because stage-2 diversity within a cluster grows with cluster
    size (larger pool → richer evaluate draws), not independently.

    Parameters
    ----------
    generator    : DirectTCVAE instance (uses sample_joint internally)
    J_candidates : list of J values to test
                   default: [25, 50, 100, 200, 300, 400, 600, 800]
    n_pool       : number of scenarios to sample for the analysis (default 10,000)
    seed         : RNG seed
    plot         : if True, display the elbow plot
    save_path    : if given, save the plot to this path

    Returns
    -------
    dict with keys:
      'J_values'  : list of J values tested
      'errors'    : corresponding mean quantization errors
      'optimal_J' : recommended J at the elbow (Kneedle method)
    """
    if J_candidates is None:
        J_candidates = [25, 50, 100, 200, 300, 400, 600, 800]

    np.random.seed(seed)
    print(f"[find_optimal] Sampling {n_pool} scenarios from generator...")
    ret1, _ = generator.sample_joint(n_pool)

    errors = []
    print(f"[find_optimal] Testing J = {J_candidates}")
    for J in J_candidates:
        if J >= n_pool:
            print(f"  J={J:4d}: skipped (J >= n_pool={n_pool})")
            errors.append(float("nan"))
            continue
        labels, medoid_idx, _ = _kmedoids_with_voronoi_probs(ret1, J, seed=seed)
        medoids = ret1[medoid_idx]                                # (J, Q)
        dists   = np.linalg.norm(ret1 - medoids[labels], axis=1) # (n_pool,)
        errors.append(float(dists.mean()))
        print(f"  J={J:4d}: mean quantization error = {errors[-1]:.6f}")

    # ── Elbow detection (Kneedle method) ─────────────────────────────────────
    valid_pairs = [(j, e) for j, e in zip(J_candidates, errors)
                   if not (isinstance(e, float) and e != e)]  # drop NaN
    J_vals  = [v[0] for v in valid_pairs]
    err_arr = np.array([v[1] for v in valid_pairs])

    # Normalize both axes to [0, 1] so the perpendicular-distance calculation
    # is not dominated by the scale of J vs. the scale of the error.
    x = np.array(J_vals, dtype=float)
    xn = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    yn = (err_arr - err_arr.min()) / max(err_arr.max() - err_arr.min(), 1e-12)

    p1  = np.array([xn[0],  yn[0]])
    p2  = np.array([xn[-1], yn[-1]])
    lv  = p2 - p1
    ll  = np.linalg.norm(lv)
    perp = [abs(np.cross(lv, p1 - np.array([xi, yi]))) / ll
            for xi, yi in zip(xn, yn)]

    optimal_J = J_vals[int(np.argmax(perp))]
    print(f"\n[find_optimal] Recommended J (elbow): {optimal_J}")
    print(f"[find_optimal] Note: evaluate nodes E=10-20 is typically sufficient; "
          f"stage-2 diversity comes from the cluster pool size, not E directly.")

    if plot or save_path:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(figsize=(8, 5))
        ax.plot(J_vals, err_arr, "o-", linewidth=2, markersize=6)
        ax.axvline(optimal_J, color="red", linestyle="--",
                   label=f"Elbow: J = {optimal_J}")
        ax.set_xlabel("Number of recourse nodes (J)")
        ax.set_ylabel("Mean quantization error (Wasserstein-1 proxy)")
        ax.set_title("Optimal recourse node count — Wasserstein elbow")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[find_optimal] Plot saved to '{save_path}'")
        if plot:
            plt.show()
        plt.close()

    return {"J_values": J_vals, "errors": err_arr.tolist(), "optimal_J": optimal_J}


# ── CLI entry point ───────────────────────────────────────────────────────────

def _cli():
    import argparse
    import sys
    import types
    import yaml
    import torch

    # Ensure both src/ and the project root are importable regardless of CWD.
    _src_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _root_dir = os.path.abspath(os.path.join(_src_dir, ".."))
    for _p in (_src_dir, _root_dir):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    parser = argparse.ArgumentParser(
        description="Export a two-stage scenario tree to CSV for the C++ CVaR solver."
    )
    parser.add_argument("--model-dir",    required=True,
                        help="Path to a final_model/ or checkpoint_epoch_N/ folder")
    parser.add_argument("--output",       default="data/scenarios/tcvae_tree.csv",
                        help="Destination CSV path")
    parser.add_argument("--generator",    default="direct",
                        choices=["direct", "vine"],
                        help="'direct' = raw TC-VAE, 'vine' = TC-VAE marginals + vine copula")
    parser.add_argument("--recourse",     type=int, default=400,
                        help="Number of recourse nodes J")
    parser.add_argument("--evaluate",     type=int, default=40,
                        help="Evaluate children per recourse node E")
    parser.add_argument("--oversampling", type=int, default=5,
                        help="Pool = J * oversampling * E (for K-medoids)")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--data-dir",     default="data/raw")
    parser.add_argument("--train-end",    type=int, default=521,
                        help="Row index of first out-of-sample week")
    parser.add_argument("--pool-size",    type=int, default=3000,
                        help="TC-VAE pool size for vine copula (--generator vine only)")
    parser.add_argument("--neighbours",   type=int, default=300,
                        help="k-NN neighbours for stage-2 conditioning (vine only)")
    parser.add_argument("--trunc-level",  type=int, default=5,
                        help="Vine truncation level (vine only)")
    parser.add_argument("--find-optimal", action="store_true",
                        help="Run Wasserstein elbow analysis to recommend J")
    parser.add_argument("--no-reduction", dest="use_reduction", action="store_false",
                        default=True,
                        help=(
                            "Skip K-medoids: sort J*E scenarios by stage-1 market return, "
                            "assign consecutive groups of E to each recourse node, uniform "
                            "p_j=1/J.  Faster and preferred when J*E >= 10,000.  "
                            "Oversampling is ignored when this flag is set."
                        ))
    vae_group = parser.add_mutually_exclusive_group()
    vae_group.add_argument("--use-vae", dest="use_vae", action="store_true", default=None,
                           help="Force VAE sampling (skip auto-detect)")
    vae_group.add_argument("--no-vae",  dest="use_vae", action="store_false",
                           help="Force historical bootstrap fallback")
    args = parser.parse_args()

    from portfolio_scenarios.data_pipeline.fetcher import load_raw_data
    from portfolio_scenarios.data_pipeline.preprocess import prep_condition_vector
    from portfolio_scenarios.data_pipeline.dataset import TCVAEDataset
    from tsvae.models.network_pipeline import NetworkPipeline
    from portfolio_scenarios.scenario_generation.generators import DirectTCVAE, TCVAEVineCopula

    checkpoint_dir = os.path.abspath(args.model_dir)
    project_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # ── Load VAE config ───────────────────────────────────────────────────────
    # The checkpoint's own exp_config.yaml always takes priority — it records
    # the exact architecture used during training.  Fall back to the canonical
    # configs/vae_config.yaml only when no exp_config.yaml is present.
    # exp_config.yaml may use a custom YAML tag (!!python/object/new:...) that
    # safe_load rejects; _load_yaml_config() handles both formats.
    def _load_yaml_config(path: str):
        """Return a plain dict from a YAML file, handling custom Python tags."""
        import re
        with open(path) as f:
            content = f.read()
        try:
            raw = yaml.safe_load(content)
            if isinstance(raw, dict):
                return raw
        except yaml.YAMLError:
            pass
        # Strip !!python/object/new:... tag and parse the dictitems block
        stripped = re.sub(r"^!!python/object/new:[^\n]+\n", "", content,
                          flags=re.MULTILINE)
        raw = yaml.safe_load(stripped)
        if isinstance(raw, dict) and "dictitems" in raw:
            return raw["dictitems"]
        return raw

    candidates = [
        os.path.join(os.path.dirname(checkpoint_dir), "exp_config.yaml"),
        os.path.join(project_root, "configs", "vae_config.yaml"),
    ]
    raw = None
    for path in candidates:
        if os.path.exists(path):
            raw = _load_yaml_config(path)
            if isinstance(raw, dict):
                print(f"[tcvae_csv] Config loaded from {path}")
                break
            raw = None
    if raw is None:
        raise FileNotFoundError(
            "Could not find a readable config. Tried:\n" +
            "\n".join(f"  {p}" for p in candidates)
        )
    exp_config = types.SimpleNamespace(**raw)

    # ── Load data ─────────────────────────────────────────────────────────────
    cfg_dataset = getattr(exp_config, "dataset", "DOW30VIX")
    _name_map   = {"DOW30": "DOW", "DOW": "DOW", "SP500": "SP100", "SP100": "SP100"}
    stock_stem  = _name_map.get(cfg_dataset.split("VIX")[0], cfg_dataset.split("VIX")[0])
    stock_file  = stock_stem + ".npz"

    data_dir = os.path.join(project_root, args.data_dir)
    tickers, prices, vix_raw = load_raw_data(data_dir, stock_file, "VIX.npz")
    T          = prices.shape[0]
    conditions = prep_condition_vector(vix_raw.ravel(), T)   # (T, 1)
    split      = min(args.train_end, T)

    train_ds = TCVAEDataset(
        prices=prices[:split], conditions=conditions[:split],
        window_size=exp_config.data_length,
    )
    hist_windows = train_ds.data.numpy()    # (N, W, Q) normalised
    hist_conds   = train_ds.labels.numpy()  # (N, 1)

    # Initial prices = price at the split boundary (first out-of-sample week).
    # This is 2024-12-26 for the default split=521, matching the optimizer's
    # initial_prices = prices.iloc[-1].  Using prices[split-1] (one week earlier,
    # 2024-12-19) inflated scenario prices by 3-5% for tech stocks during a down
    # week, causing unrealistic 6%/week max target returns in get_mu_range().
    # Clamped to the last available row in case the DOW.npz is ever extended.
    initial_prices = prices[min(split, len(prices) - 1)].astype(np.float64)  # (Q,)

    tickers = list(tickers)
    print(f"[tcvae_csv] {len(tickers)} assets, split at row {split}, "
          f"initial prices from row {split - 1}")

    # ── Build and load model ──────────────────────────────────────────────────
    model = NetworkPipeline()(exp_config)
    ckpt_path = os.path.join(checkpoint_dir, "model.pt")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    print(f"[tcvae_csv] Model loaded from {ckpt_path}")

    # ── Build generator ───────────────────────────────────────────────────────
    tcvae_gen = DirectTCVAE(
        tcvae_model=model,
        historical_data=hist_windows,
        historical_conditions=hist_conds,
    )

    if args.generator == "vine":
        hist_ret1 = hist_windows[:, 1, :] - 1.0   # week-1 simple returns for vine fitting
        generator = TCVAEVineCopula(
            tcvae_generator=tcvae_gen,
            historical_returns=hist_ret1,
            pool_size=args.pool_size,
            n_neighbours=args.neighbours,
            truncation_level=args.trunc_level,
            seed=args.seed,
        )
    else:
        generator = tcvae_gen

    # ── Optional elbow analysis ───────────────────────────────────────────────
    if args.find_optimal:
        elbow_path = os.path.join(project_root, "results", "elbow_analysis.png")
        find_optimal_tree_size(
            generator=generator,
            seed=args.seed,
            plot=False,
            save_path=elbow_path,
        )

    build_tcvae_tree_csv(
        generator      = generator,
        initial_prices = initial_prices,
        num_recourse   = args.recourse,
        num_evaluate   = args.evaluate,
        output_path    = args.output,
        oversampling   = args.oversampling,
        seed           = args.seed,
        use_vae        = args.use_vae,
        use_reduction  = args.use_reduction,
    )


if __name__ == "__main__":
    _cli()
