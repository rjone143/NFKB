"""
residence_nfkb_v2.py

Fixes two issues in the original residence_nfkb.py:

1. POOLING: the original script averaged across ALL cells and ALL replicates
   into a single trajectory before extracting dwell times, which is why only
   3 dwell segments were found. This version extracts dwell times PER CELL
   (using each cell's own threshold-crossing behavior) and pools across
   cells, using the full available sample size.

2. NULL DISTRIBUTION: the original script compared against a SINGLE random
   draw from a geometric distribution (size = len(res_times), i.e. as few as
   3 samples) -- not a proper null. This version draws the geometric null
   MANY times (default 10,000) and reports an empirical p-value / confidence
   interval, not a single noisy point estimate.

Run this after nfkb_load_percell.py is in the same directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from nfkb_load_percell import load_percell_matrix

MAT_FILE = "AllDataSets.mat"
CONDITION_INDEX = 0  # 0 = LPS_R1
DOWNSAMPLE = 10
N_NULL_DRAWS = 10000
TAIL_PERCENTILE = 75


def extract_dwell_times_percell(X, threshold=None):
    """
    Extract dwell times from EACH cell's own trajectory and pool them.
    threshold: if None, uses each cell's own temporal mean (matches the
               original script's thresholding logic, applied per cell
               instead of to one pooled trajectory).
    Returns: pooled dwell times (all cells combined), and a per-cell list
             (for optional mixed-effects / clustering-aware analysis).
    """
    all_dwells = []
    per_cell_dwells = []
    for i in range(X.shape[0]):
        row = X[i]
        thresh = np.nanmean(row) if threshold is None else threshold
        state = (row > thresh).astype(int)
        res_times = []
        count = 1
        for t in range(1, len(state)):
            if state[t] == state[t - 1]:
                count += 1
            else:
                res_times.append(count)
                count = 1
        res_times.append(count)
        per_cell_dwells.append(np.array(res_times))
        all_dwells.extend(res_times)
    return np.array(all_dwells), per_cell_dwells


def monte_carlo_geometric_null(observed_dwells, tail_threshold, n_draws=N_NULL_DRAWS, seed=0):
    """
    Proper null: draw n_draws independent geometric samples (matched in
    sample size and mean dwell time to the observed data) and build an
    empirical distribution of the tail ratio, rather than relying on one draw.
    """
    rng = np.random.default_rng(seed)
    n = len(observed_dwells)
    p = 1.0 / np.mean(observed_dwells)
    p = min(max(p, 1e-6), 0.999)

    null_tail_ratios = np.empty(n_draws)
    null_means = np.empty(n_draws)
    for k in range(n_draws):
        draw = rng.geometric(p=p, size=n)
        null_tail_ratios[k] = np.mean(draw >= tail_threshold)
        null_means[k] = np.mean(draw)

    return null_tail_ratios, null_means


def main():
    X, replicate_id, rep_name = load_percell_matrix(
        MAT_FILE, CONDITION_INDEX, downsample=DOWNSAMPLE
    )

    pooled_dwells, per_cell_dwells = extract_dwell_times_percell(X)
    print(f"\nTotal dwell segments (pooled across {X.shape[0]} cells): {len(pooled_dwells)}")
    print(f"Mean dwell time: {np.mean(pooled_dwells):.3f}")
    print(f"Median dwell time: {np.median(pooled_dwells):.3f}")

    tail_threshold = np.percentile(pooled_dwells, TAIL_PERCENTILE)
    tail_ratio_data = np.mean(pooled_dwells >= tail_threshold)
    print(f"Tail threshold (P{TAIL_PERCENTILE}): {tail_threshold:.3f}")
    print(f"Observed tail ratio: {tail_ratio_data:.4f}")

    null_tail_ratios, null_means = monte_carlo_geometric_null(
        pooled_dwells, tail_threshold, n_draws=N_NULL_DRAWS
    )
    null_ci = np.percentile(null_tail_ratios, [2.5, 97.5])
    p_value = float(np.mean(null_tail_ratios >= tail_ratio_data))

    print(f"\nMonte Carlo null (n={N_NULL_DRAWS} draws):")
    print(f"  Null tail ratio mean: {null_tail_ratios.mean():.4f}")
    print(f"  Null tail ratio 95% CI: [{null_ci[0]:.4f}, {null_ci[1]:.4f}]")
    print(f"  Observed tail ratio: {tail_ratio_data:.4f}")
    print(f"  Empirical p-value (observed >= null): {p_value:.4f}")
    print("  (p < 0.05 means observed persistence exceeds the memoryless "
          "null more than chance would predict)")

    # Plot: pooled empirical CCDF vs null distribution's CCDF envelope
    x_sorted = np.sort(pooled_dwells)
    y_ccdf = 1.0 - np.arange(1, len(x_sorted) + 1) / len(x_sorted)

    plt.figure(figsize=(7, 5))
    plt.plot(x_sorted, y_ccdf, "o-", label=f"NF-kB (pooled, n={len(pooled_dwells)})")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Dwell time")
    plt.ylabel("CCDF")
    plt.title(f"Pooled per-cell dwell-time CCDF: {rep_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("nfkb_v2_dwell_ccdf_percell.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(null_tail_ratios, bins=50, alpha=0.7, label="Monte Carlo null")
    plt.axvline(tail_ratio_data, color="red", linestyle="--", label="Observed")
    plt.xlabel("Tail ratio")
    plt.ylabel("Count")
    plt.title(f"Null distribution vs. observed tail ratio: {rep_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("nfkb_v2_null_distribution.png", dpi=200)
    plt.close()

    print("\nSaved: nfkb_v2_dwell_ccdf_percell.png, nfkb_v2_null_distribution.png")


if __name__ == "__main__":
    main()
