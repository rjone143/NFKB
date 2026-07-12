"""
NF-kB / signaling dynamics analysis pipeline
--------------------------------------------

This script analyzes single-cell signaling trajectories for:
1. Pre-commitment coherence (mutual information over time)
2. Activated-state dwell-time distributions
3. Phase-sensitive response differences

Input:
    CSV file with columns:
        - cell_id
        - time
        - signal
        - outcome
    Optional columns:
        - stimulus_time
        - commitment_time

Outputs:
    - Pre-commitment coherence plot
    - Dwell-time CCDF + model fits
    - Phase-bin outcome comparison plot
    - Summary table printed to console

Author-ready notes:
    - Replace file path in CONFIG
    - Tune thresholds in CONFIG
    - Check that outcome encoding matches your dataset
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer


# ----------------------------
# CONFIG
# ----------------------------

CONFIG = {
    "csv_path": "nfkb_timeseries.csv",
    "time_col": "time",
    "cell_col": "cell_id",
    "signal_col": "signal",
    "outcome_col": "outcome",
    "stimulus_time_col": "stimulus_time",      # optional
    "commitment_time_col": "commitment_time",  # optional

    # Commitment estimation fallback if commitment_time is absent
    "commit_threshold": 0.60,
    "commit_min_duration": 15.0,   # same unit as time
    "activation_threshold": 0.50,

    # MI settings
    "mi_window_width": 20.0,
    "mi_step": 5.0,
    "feature_bins": 5,
    "n_permutations": 200,

    # Phase analysis
    "n_phase_bins": 6,

    # Plot/output
    "save_prefix": "nfkb_analysis",
}


# ----------------------------
# UTILITIES
# ----------------------------

def validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def encode_outcomes(series: pd.Series) -> np.ndarray:
    """
    Encode categorical or numeric outcome labels into integers.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy()
    codes, _ = pd.factorize(series.astype(str))
    return codes


def compute_time_step(times: np.ndarray) -> float:
    """
    Estimate the typical time step from sorted unique times.
    """
    unique_times = np.sort(np.unique(times))
    if len(unique_times) < 2:
        raise ValueError("Need at least two unique time values.")
    diffs = np.diff(unique_times)
    return float(np.median(diffs))


def first_commitment_time(
    cell_df: pd.DataFrame,
    signal_col: str,
    time_col: str,
    threshold: float,
    min_duration: float
) -> Optional[float]:
    """
    Estimate commitment time as the first time the signal exceeds threshold
    and stays above threshold for at least min_duration.
    """
    sdf = cell_df.sort_values(time_col).copy()
    times = sdf[time_col].to_numpy()
    signal = sdf[signal_col].to_numpy()

    if len(times) < 2:
        return None

    dt = compute_time_step(times)
    min_points = max(1, int(math.ceil(min_duration / dt)))

    above = signal >= threshold
    run_start = None
    run_len = 0

    for i, val in enumerate(above):
        if val:
            if run_start is None:
                run_start = i
                run_len = 1
            else:
                run_len += 1
            if run_len >= min_points:
                return float(times[run_start])
        else:
            run_start = None
            run_len = 0

    return None


def add_commitment_times(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Ensure each cell has a commitment_time.
    """
    commit_col = cfg["commitment_time_col"]
    cell_col = cfg["cell_col"]
    time_col = cfg["time_col"]
    signal_col = cfg["signal_col"]

    if commit_col in df.columns and df[commit_col].notna().any():
        return df.copy()

    commit_times = {}
    for cell_id, g in df.groupby(cell_col):
        ct = first_commitment_time(
            g,
            signal_col=signal_col,
            time_col=time_col,
            threshold=cfg["commit_threshold"],
            min_duration=cfg["commit_min_duration"]
        )
        commit_times[cell_id] = ct

    out = df.copy()
    out[commit_col] = out[cell_col].map(commit_times)
    return out


def summarize_window_features(
    window_df: pd.DataFrame,
    cfg: Dict
) -> pd.DataFrame:
    """
    Compute per-cell features inside a time window.
    """
    cell_col = cfg["cell_col"]
    signal_col = cfg["signal_col"]

    features = []
    for cell_id, g in window_df.groupby(cell_col):
        signal = g[signal_col].to_numpy()
        if len(signal) == 0:
            continue

        feat = {
            cell_col: cell_id,
            "mean_signal": float(np.mean(signal)),
            "max_signal": float(np.max(signal)),
            "std_signal": float(np.std(signal, ddof=0)),
            "auc_signal": float(np.trapezoid(signal, x=g[cfg["time_col"]].to_numpy()))        }
        features.append(feat)

    return pd.DataFrame(features)


def discretize_feature(x: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Discretize a continuous feature for MI computation.
    """
    x = np.asarray(x).reshape(-1, 1)
    est = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return est.fit_transform(x).astype(int).ravel()


def permutation_pvalue(observed: float, x_disc: np.ndarray, y: np.ndarray, n_perm: int) -> float:
    """
    Permutation p-value for mutual information.
    """
    null_vals = []
    rng = np.random.default_rng(42)

    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        null_vals.append(mutual_info_score(x_disc, y_perm))

    null_vals = np.asarray(null_vals)
    p = (np.sum(null_vals >= observed) + 1) / (len(null_vals) + 1)
    return float(p)


# ----------------------------
# 1. PRE-COMMITMENT COHERENCE
# ----------------------------

@dataclass
class MIRow:
    time_center: float
    mi: float
    p_value: float
    n_cells: int


def compute_precommitment_mi(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Compute mutual information between windowed trajectory features and outcome
    as a function of time relative to commitment.
    """
    time_col = cfg["time_col"]
    cell_col = cfg["cell_col"]
    outcome_col = cfg["outcome_col"]
    commit_col = cfg["commitment_time_col"]

    valid_cells = df.groupby(cell_col)[commit_col].first().dropna().index
    dfx = df[df[cell_col].isin(valid_cells)].copy()

    if dfx.empty:
        raise ValueError("No valid cells with commitment times available.")

    # Align time to commitment for each cell
    commit_map = dfx.groupby(cell_col)[commit_col].first().to_dict()
    dfx["time_to_commit"] = dfx.apply(
        lambda r: r[time_col] - commit_map[r[cell_col]], axis=1
    )

    tmin = float(dfx["time_to_commit"].min())
    tmax = float(dfx["time_to_commit"].max())
    width = cfg["mi_window_width"]
    step = cfg["mi_step"]

    rows: List[MIRow] = []

    for start in np.arange(tmin, tmax - width + step, step):
        end = start + width
        center = start + width / 2.0

        win = dfx[(dfx["time_to_commit"] >= start) & (dfx["time_to_commit"] < end)]
        feats = summarize_window_features(win, cfg)

        if feats.empty:
            continue

        outcomes = (
            dfx[[cell_col, outcome_col]]
            .drop_duplicates()
            .set_index(cell_col)
            .loc[feats[cell_col], outcome_col]
        )
        y = encode_outcomes(outcomes)

        # Use max_signal as primary feature; can expand later
        x = feats["max_signal"].to_numpy()

        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            continue

        x_disc = discretize_feature(x, cfg["feature_bins"])
        mi = mutual_info_score(x_disc, y)
        p = permutation_pvalue(mi, x_disc, y, cfg["n_permutations"])

        rows.append(MIRow(
            time_center=float(center),
            mi=float(mi),
            p_value=float(p),
            n_cells=int(len(feats))
        ))

    return pd.DataFrame([r.__dict__ for r in rows])


def plot_precommitment_mi(mi_df: pd.DataFrame, cfg: Dict) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(mi_df["time_center"], mi_df["mi"], marker="o")
    plt.axvline(0, linestyle="--")
    plt.xlabel("Time relative to commitment")
    plt.ylabel("Mutual information")
    plt.title("Pre-commitment coherence transient")
    plt.tight_layout()
    plt.savefig(f"{cfg['save_prefix']}_precommitment_mi.png", dpi=200)
    plt.close()


# ----------------------------
# 2. DWELL-TIME ANALYSIS
# ----------------------------

def extract_dwell_times(df: pd.DataFrame, cfg: Dict) -> np.ndarray:
    """
    Extract activated-state dwell times across all cells using activation_threshold.
    """
    time_col = cfg["time_col"]
    cell_col = cfg["cell_col"]
    signal_col = cfg["signal_col"]
    threshold = cfg["activation_threshold"]

    dwell_times = []

    for _, g in df.groupby(cell_col):
        g = g.sort_values(time_col)
        times = g[time_col].to_numpy()
        signal = g[signal_col].to_numpy()

        if len(times) < 2:
            continue

        dt = compute_time_step(times)
        active = signal >= threshold

        run_len = 0
        for val in active:
            if val:
                run_len += 1
            else:
                if run_len > 0:
                    dwell_times.append(run_len * dt)
                    run_len = 0
        if run_len > 0:
            dwell_times.append(run_len * dt)

    return np.asarray(dwell_times, dtype=float)


def fit_exponential(data: np.ndarray) -> Dict:
    loc, scale = stats.expon.fit(data, floc=0)
    ll = np.sum(stats.expon.logpdf(data, loc=loc, scale=scale))
    k = 1
    aic = 2 * k - 2 * ll
    return {"model": "exponential", "params": {"scale": scale}, "loglik": ll, "aic": aic}


def fit_lognormal(data: np.ndarray) -> Dict:
    shape, loc, scale = stats.lognorm.fit(data, floc=0)
    ll = np.sum(stats.lognorm.logpdf(data, s=shape, loc=loc, scale=scale))
    k = 2
    aic = 2 * k - 2 * ll
    return {"model": "lognormal", "params": {"shape": shape, "scale": scale}, "loglik": ll, "aic": aic}


def fit_weibull(data: np.ndarray) -> Dict:
    c, loc, scale = stats.weibull_min.fit(data, floc=0)
    ll = np.sum(stats.weibull_min.logpdf(data, c=c, loc=loc, scale=scale))
    k = 2
    aic = 2 * k - 2 * ll
    return {"model": "weibull", "params": {"c": c, "scale": scale}, "loglik": ll, "aic": aic}


def model_comparison_table(dwell_times: np.ndarray) -> pd.DataFrame:
    fits = [
        fit_exponential(dwell_times),
        fit_lognormal(dwell_times),
        fit_weibull(dwell_times),
    ]
    table = pd.DataFrame(fits)
    table = table.sort_values("aic").reset_index(drop=True)
    table["delta_aic"] = table["aic"] - table["aic"].min()
    return table


def ccdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(data)
    y = 1.0 - np.arange(1, len(x) + 1) / len(x)
    return x, y


def plot_dwell_ccdf(dwell_times: np.ndarray, table: pd.DataFrame, cfg: Dict) -> None:
    x, y = ccdf(dwell_times)

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o", linestyle="none", label="Empirical CCDF")

    xs = np.linspace(max(np.min(dwell_times), 1e-6), np.max(dwell_times), 300)

    for _, row in table.iterrows():
        model = row["model"]
        params = row["params"]

        if model == "exponential":
            sf = stats.expon.sf(xs, loc=0, scale=params["scale"])
        elif model == "lognormal":
            sf = stats.lognorm.sf(xs, s=params["shape"], loc=0, scale=params["scale"])
        elif model == "weibull":
            sf = stats.weibull_min.sf(xs, c=params["c"], loc=0, scale=params["scale"])
        else:
            continue

        plt.plot(xs, sf, label=model)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Activated-state dwell time")
    plt.ylabel("CCDF")
    plt.title("Activated-state dwell-time distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg['save_prefix']}_dwell_ccdf.png", dpi=200)
    plt.close()


# ----------------------------
# 3. PHASE-SENSITIVE RESPONSE
# ----------------------------

def compute_phase_bin_outcomes(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Approximate phase sensitivity by binning cells according to the phase of their
    signaling trajectory at stimulus time.

    If stimulus_time is missing, falls back to the first recorded timepoint.
    """
    time_col = cfg["time_col"]
    cell_col = cfg["cell_col"]
    signal_col = cfg["signal_col"]
    outcome_col = cfg["outcome_col"]
    stim_col = cfg["stimulus_time_col"]

    rows = []

    for cell_id, g in df.groupby(cell_col):
        g = g.sort_values(time_col)
        times = g[time_col].to_numpy()
        signal = g[signal_col].to_numpy()

        if len(times) < 3:
            continue

        stimulus_time = None
        if stim_col in g.columns and g[stim_col].notna().any():
            stimulus_time = float(g[stim_col].dropna().iloc[0])
        else:
            stimulus_time = float(times[0])

        idx = int(np.argmin(np.abs(times - stimulus_time)))

        # Use local derivative as rough phase proxy
        if idx == 0:
            deriv = signal[1] - signal[0]
        elif idx == len(signal) - 1:
            deriv = signal[-1] - signal[-2]
        else:
            deriv = (signal[idx + 1] - signal[idx - 1]) / 2.0

        amp = signal[idx]
        phase_proxy = math.atan2(deriv, amp - np.mean(signal))

        outcome = g[outcome_col].iloc[0]

        rows.append({
            cell_col: cell_id,
            "phase_proxy": phase_proxy,
            outcome_col: outcome
        })

    phase_df = pd.DataFrame(rows)
    if phase_df.empty:
        return phase_df

    phase_df["phase_bin"] = pd.cut(
        phase_df["phase_proxy"],
        bins=cfg["n_phase_bins"],
        labels=False,
        include_lowest=True
    )
    return phase_df


def phase_outcome_test(phase_df: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, float]:
    """
    Chi-square test for phase-bin dependence of outcome.
    """
    if phase_df.empty:
        return pd.DataFrame(), np.nan

    contingency = pd.crosstab(phase_df["phase_bin"], phase_df[cfg["outcome_col"]])
    chi2, p, _, _ = stats.chi2_contingency(contingency)
    return contingency, float(p)


def plot_phase_outcomes(contingency: pd.DataFrame, cfg: Dict) -> None:
    if contingency.empty:
        return

    prop = contingency.div(contingency.sum(axis=1), axis=0)
    prop.plot(kind="bar", stacked=True, figsize=(8, 5))
    plt.xlabel("Phase bin")
    plt.ylabel("Outcome proportion")
    plt.title("Phase-sensitive outcome structure")
    plt.tight_layout()
    plt.savefig(f"{cfg['save_prefix']}_phase_outcomes.png", dpi=200)
    plt.close()


# ----------------------------
# MAIN
# ----------------------------

def main() -> None:
    # Load data
    df = pd.read_csv(CONFIG["csv_path"])

    required = [
        CONFIG["cell_col"],
        CONFIG["time_col"],
        CONFIG["signal_col"],
        CONFIG["outcome_col"],
    ]
    validate_columns(df, required)

    # Add commitment times if absent
    df = add_commitment_times(df, CONFIG)

    print("\n=== DATA SUMMARY ===")
    print(f"Rows: {len(df)}")
    print(f"Cells: {df[CONFIG['cell_col']].nunique()}")
    print(f"Outcomes: {df[CONFIG['outcome_col']].nunique()}")

    # 1. Pre-commitment MI
    print("\n=== PRE-COMMITMENT COHERENCE ===")
    mi_df = compute_precommitment_mi(df, CONFIG)
    print(mi_df.head())
    if not mi_df.empty:
        peak_row = mi_df.loc[mi_df["mi"].idxmax()]
        print(
            f"Peak MI at time {peak_row['time_center']:.2f}, "
            f"MI={peak_row['mi']:.4f}, p={peak_row['p_value']:.4f}, "
            f"n_cells={int(peak_row['n_cells'])}"
        )
        plot_precommitment_mi(mi_df, CONFIG)
    else:
        print("No MI results computed.")

    # 2. Dwell times
    print("\n=== DWELL-TIME MODEL COMPARISON ===")
    dwell_times = extract_dwell_times(df, CONFIG)
    if len(dwell_times) > 0:
        table = model_comparison_table(dwell_times)
        print(table[["model", "loglik", "aic", "delta_aic"]])
        plot_dwell_ccdf(dwell_times, table, CONFIG)
    else:
        print("No dwell times extracted.")
        table = pd.DataFrame()

    # 3. Phase sensitivity
    print("\n=== PHASE-SENSITIVE RESPONSE ===")
    phase_df = compute_phase_bin_outcomes(df, CONFIG)
    contingency, phase_p = phase_outcome_test(phase_df, CONFIG)
    if not contingency.empty:
        print(contingency)
        print(f"Chi-square phase dependence p-value: {phase_p:.4f}")
        plot_phase_outcomes(contingency, CONFIG)
    else:
        print("No phase analysis results computed.")

    # Save summary tables
    if not mi_df.empty:
        mi_df.to_csv(f"{CONFIG['save_prefix']}_precommitment_mi.csv", index=False)
    if len(dwell_times) > 0:
        pd.DataFrame({"dwell_time": dwell_times}).to_csv(
            f"{CONFIG['save_prefix']}_dwell_times.csv", index=False
        )
        table.to_csv(f"{CONFIG['save_prefix']}_dwell_model_table.csv", index=False)
    if not phase_df.empty:
        phase_df.to_csv(f"{CONFIG['save_prefix']}_phase_bins.csv", index=False)

    print("\nAnalysis complete.")
    print("Saved outputs:")
    print(f" - {CONFIG['save_prefix']}_precommitment_mi.png")
    print(f" - {CONFIG['save_prefix']}_dwell_ccdf.png")
    print(f" - {CONFIG['save_prefix']}_phase_outcomes.png")


if __name__ == "__main__":
    main()
