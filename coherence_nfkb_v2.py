"""
coherence_nfkb_v2.py

Improves on the original coherence_nfkb.py in two ways:

1. SAMPLE SIZE: uses per-cell trajectories (potentially hundreds of cells)
   instead of collapsing to a handful of per-replicate means before computing
   cross-trajectory variance. This gives a much less noisy coherence estimate.

2. SIGNIFICANCE TEST: the original script just reported the peak coherence
   index with no test of whether it's distinguishable from chance. This
   version adds a circular-shift permutation null: each cell's trajectory is
   independently, randomly time-shifted (circularly, so no data is lost),
   which destroys genuine cross-cell temporal alignment while preserving each
   cell's own trajectory shape and variance. If real transient synchrony
   exists, the observed peak coherence should exceed this null.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from nfkb_load_percell import load_percell_matrix

MAT_FILE = "AllDataSets.mat"
CONDITION_INDEX = 0  # 0 = LPS_R1
DOWNSAMPLE = 10
SMOOTHING_SIGMA = 2.0  # disclose this explicitly -- it affects peak location
N_PERMUTATIONS = 1000


def compute_coherence(X, smoothing_sigma=SMOOTHING_SIGMA):
    """X: (n_cells, n_timepoints). Returns smoothed coherence curve."""
    cross_var = np.var(X, axis=0)
    coherence = 1.0 / (cross_var + 1e-6)
    coherence = gaussian_filter1d(coherence, sigma=smoothing_sigma)
    return coherence


def circular_shift_null(X, n_perm=N_PERMUTATIONS, smoothing_sigma=SMOOTHING_SIGMA, seed=0):
    """
    Null distribution for peak coherence: independently circularly shift each
    cell's trajectory by a random offset, recompute coherence, record the max.
    """
    rng = np.random.default_rng(seed)
    n_cells, n_time = X.shape
    peak_null = np.empty(n_perm)
    for p in range(n_perm):
        shifts = rng.integers(0, n_time, size=n_cells)
        X_shifted = np.array([np.roll(X[i], shifts[i]) for i in range(n_cells)])
        coh = compute_coherence(X_shifted, smoothing_sigma)
        peak_null[p] = coh.max()
    return peak_null


def main():
    X, replicate_id, rep_name = load_percell_matrix(
        MAT_FILE, CONDITION_INDEX, downsample=DOWNSAMPLE
    )

    coherence = compute_coherence(X)
    peak_idx = int(np.argmax(coherence))
    peak_val = float(coherence[peak_idx])

    print(f"\nPeak coherence index: {peak_idx} (of {len(coherence)} timepoints)")
    print(f"Peak coherence value: {peak_val:.4f}")
    print(f"(smoothing sigma = {SMOOTHING_SIGMA}; disclose this in methods -- "
          f"it shifts the apparent peak location)")

    print(f"\nRunning circular-shift permutation null (n={N_PERMUTATIONS})...")
    null_peaks = circular_shift_null(X, n_perm=N_PERMUTATIONS)
    p_value = float(np.mean(null_peaks >= peak_val))
    print(f"Null peak coherence: mean={null_peaks.mean():.4f}, "
          f"95th pct={np.percentile(null_peaks, 95):.4f}")
    print(f"Observed peak coherence: {peak_val:.4f}")
    print(f"Empirical p-value (observed >= null): {p_value:.4f}")
    print("  (p < 0.05 means the coherence transient is unlikely to arise "
          "just from each cell's own pulse shape without real cross-cell "
          "temporal alignment)")

    plt.figure(figsize=(8, 4))
    plt.plot(coherence, label="Observed")
    plt.axvline(peak_idx, color="red", linestyle="--", label=f"Peak (t={peak_idx})")
    plt.title(f"NF-kB Coherence (per-cell, n={X.shape[0]}): {rep_name}")
    plt.xlabel("Time")
    plt.ylabel("1 / variance (smoothed)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("nfkb_v2_coherence.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(null_peaks, bins=50, alpha=0.7, label="Circular-shift null")
    plt.axvline(peak_val, color="red", linestyle="--", label="Observed peak")
    plt.xlabel("Peak coherence")
    plt.ylabel("Count")
    plt.title(f"Peak coherence: observed vs. null: {rep_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("nfkb_v2_coherence_null.png", dpi=200)
    plt.close()

    print("\nSaved: nfkb_v2_coherence.png, nfkb_v2_coherence_null.png")


if __name__ == "__main__":
    main()
