"""
phase_nfkb_v2.py

The original phase_nfkb.py computed:
    response = mean(signal[t:t+2]) - mean(signal[t-2:t])
on the single pooled mean trajectory. This is a local finite-difference of an
UNPERTURBED curve -- it will be "non-flat" for any rising-then-falling pulse
and does not test phase-dependent sensitivity to a perturbation, because no
perturbation is applied anywhere in that script. It cannot support the claim
that "identical perturbations produce different outcomes depending on when
they occur."

This version offers two honestly-labeled alternatives:

METHOD 1 (diagnostic only, kept for transparency):
    "Trajectory slope profile" -- the same finite-difference computation as
    the original, but correctly labeled as a description of the pulse's own
    shape, NOT as evidence of phase-dependent response. Useful to report but
    not to claim discriminating power from.

METHOD 2 (real test, uses natural cell-to-cell variation):
    Each cell activates at a slightly different time (natural biological
    jitter). Define each cell's own "phase" as its rise time relative to the
    population mean, and test whether a genuine DOWNSTREAM feature of that
    same cell's trajectory (post-peak decay rate) varies systematically with
    that phase, using a permutation test for significance. This is
    correlational (not a controlled perturbation experiment) but it is a real
    empirical test grounded in actual data variation rather than a
    re-derivation of the mean curve's own slope.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from nfkb_load_percell import load_percell_matrix

MAT_FILE = "AllDataSets.mat"
CONDITION_INDEX = 0  # 0 = LPS_R1
DOWNSAMPLE = 10
N_PERMUTATIONS = 2000


# ---------------------------------------------------------------
# METHOD 1 -- honestly-labeled diagnostic (NOT a perturbation test)
# ---------------------------------------------------------------

def trajectory_slope_profile(mean_signal, window=2):
    responses = []
    times = range(window, len(mean_signal) - window)
    for t in times:
        pre = mean_signal[t - window:t]
        post = mean_signal[t:t + window]
        responses.append(np.mean(post) - np.mean(pre))
    return np.array(list(times)), np.array(responses)


# ---------------------------------------------------------------
# METHOD 2 -- real test using natural cell-to-cell phase variation
# ---------------------------------------------------------------

def cell_rise_time(row, frac=0.5):
    """Time index at which this cell's trajectory first reaches `frac` of its
    own peak value (a simple, defensible per-cell 'phase' marker)."""
    peak_val = np.max(row)
    peak_idx = np.argmax(row)
    if peak_idx == 0:
        return 0
    target = frac * peak_val
    for t in range(peak_idx + 1):
        if row[t] >= target:
            return t
    return peak_idx


def cell_decay_rate(row, post_window=10):
    """Simple downstream outcome feature: linear decay slope in the window
    immediately following this cell's own peak."""
    peak_idx = np.argmax(row)
    end = min(peak_idx + post_window, len(row))
    if end - peak_idx < 3:
        return np.nan
    y = row[peak_idx:end]
    x = np.arange(len(y))
    slope, _, _, _, _ = stats.linregress(x, y)
    return slope


def natural_phase_sensitivity_test(X, n_perm=N_PERMUTATIONS, seed=0):
    """
    Tests whether each cell's own rise-time phase predicts its own
    post-peak decay rate, using Spearman correlation + a permutation test.
    """
    rise_times = np.array([cell_rise_time(X[i]) for i in range(X.shape[0])])
    decay_rates = np.array([cell_decay_rate(X[i]) for i in range(X.shape[0])])

    valid = ~np.isnan(decay_rates)
    rise_times = rise_times[valid]
    decay_rates = decay_rates[valid]

    rho, p_asymptotic = stats.spearmanr(rise_times, decay_rates)

    rng = np.random.default_rng(seed)
    null_rhos = np.empty(n_perm)
    for k in range(n_perm):
        shuffled = rng.permutation(decay_rates)
        null_rhos[k], _ = stats.spearmanr(rise_times, shuffled)
    p_perm = float(np.mean(np.abs(null_rhos) >= abs(rho)))

    return rise_times, decay_rates, rho, p_perm, null_rhos


def main():
    X, replicate_id, rep_name = load_percell_matrix(
        MAT_FILE, CONDITION_INDEX, downsample=DOWNSAMPLE
    )
    mean_signal = np.mean(X, axis=0)

    # Method 1: diagnostic only
    times, responses = trajectory_slope_profile(mean_signal)
    plt.figure(figsize=(8, 4))
    plt.plot(times, responses)
    plt.axhline(0, linestyle="--")
    plt.title(f"Trajectory slope profile (diagnostic only): {rep_name}")
    plt.xlabel("Time")
    plt.ylabel("Local slope of mean trajectory")
    plt.tight_layout()
    plt.savefig("nfkb_v2_slope_profile.png", dpi=200)
    plt.close()
    print(f"\n[Method 1 -- diagnostic] slope-profile variance: {np.var(responses):.4f}")
    print("This describes the mean trajectory's own shape. It is NOT evidence")
    print("of phase-dependent response to a perturbation and should not be")
    print("reported as such.")

    # Method 2: real natural-variation test
    print(f"\n[Method 2 -- real test] natural phase-sensitivity "
          f"(n_cells={X.shape[0]})")
    rise_times, decay_rates, rho, p_perm, null_rhos = natural_phase_sensitivity_test(X)
    print(f"Spearman rho (rise-time phase vs. post-peak decay rate): {rho:.4f}")
    print(f"Permutation p-value (n={N_PERMUTATIONS}): {p_perm:.4f}")
    if p_perm < 0.05:
        print("--> Significant: a cell's activation timing predicts its own "
              "downstream decay behavior. This is a genuine (if correlational) "
              "phase-sensitivity result.")
    else:
        print("--> Not significant at alpha=0.05: no detectable relationship "
              "between activation timing and downstream decay in this dataset.")

    plt.figure(figsize=(6, 5))
    plt.scatter(rise_times, decay_rates, alpha=0.4, s=15)
    plt.xlabel("Cell rise time (phase proxy)")
    plt.ylabel("Post-peak decay rate")
    plt.title(f"Natural phase vs. outcome, rho={rho:.3f}, p={p_perm:.4f}: {rep_name}")
    plt.tight_layout()
    plt.savefig("nfkb_v2_phase_natural.png", dpi=200)
    plt.close()

    print("\nSaved: nfkb_v2_slope_profile.png, nfkb_v2_phase_natural.png")


if __name__ == "__main__":
    main()
