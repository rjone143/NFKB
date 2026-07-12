"""
nfkb_load_percell.py

Shared loader for the NF-kB reanalysis pipeline. Unlike the original scripts,
this keeps INDIVIDUAL CELL trajectories rather than collapsing to a single
per-replicate mean immediately. This is what lets the downstream analyses
(coherence, residence, phase) use the full available sample size instead of
pooling everything into one trajectory before any statistics are computed.

Usage:
    from nfkb_load_percell import load_percell_matrix
    X, replicate_id, rep_name = load_percell_matrix(
        mat_file="AllDataSets.mat",
        condition_index=0,       # 0 = LPS_R1
        signal_key="NFkBdim_Ratio",
        downsample=10,
    )
    # X: (n_cells_total, n_timepoints) array, z-scored per cell
    # replicate_id: (n_cells_total,) array labeling which "measure" entry
    #               (replicate/well/FOV) each cell came from
"""

import h5py
import numpy as np


def decode_uint16_string(arr):
    arr = np.array(arr).flatten()
    return "".join(chr(int(x)) for x in arr if int(x) != 0)


def load_percell_matrix(mat_file, condition_index=0, signal_key="NFkBdim_Ratio",
                         downsample=10, min_valid_frac=0.8):
    """
    Returns:
        X: (n_cells, n_timepoints) z-scored per-cell trajectories, downsampled
        replicate_id: (n_cells,) int array, which measure/replicate entry each
                      cell came from (use this for mixed-effects / clustering
                      corrections if you pool across replicates for stats)
        rep_name: str, decoded condition name (sanity check against condition_index)
    """
    with h5py.File(mat_file, "r") as f:
        ds = f["DataSets"]
        data_refs = ds["Data"][()]
        repname_refs = ds["RepNames"][()]

        rep_name = decode_uint16_string(f[repname_refs[condition_index, 0]][()])
        print("Condition:", rep_name)

        data_group = f[data_refs[condition_index, 0]]
        measure = data_group["measure"][()]

        all_cells = []
        replicate_id = []
        for j in range(measure.shape[0]):
            ref = measure[j, 0]
            obj = f[ref]
            nfkb = np.array(obj[signal_key][()])  # shape: time x cells (per replicate)
            # keep EVERY cell's trajectory instead of averaging over axis=1
            n_cells_this_rep = nfkb.shape[1]
            for c in range(n_cells_this_rep):
                all_cells.append(nfkb[:, c])
                replicate_id.append(j)

    X = np.array(all_cells)  # (n_cells_total, n_timepoints_raw)
    replicate_id = np.array(replicate_id)

    # Drop cells with too many missing values (NaNs) before downsampling
    valid_frac = 1.0 - np.isnan(X).mean(axis=1)
    keep = valid_frac >= min_valid_frac
    n_dropped = int((~keep).sum())
    if n_dropped > 0:
        print(f"Dropping {n_dropped} / {len(keep)} cells with >"
              f"{100*(1-min_valid_frac):.0f}% missing timepoints")
    X = X[keep]
    replicate_id = replicate_id[keep]

    # Downsample
    X = X[:, ::downsample]

    # Interpolate any remaining short NaN gaps, then z-score each cell
    for i in range(X.shape[0]):
        row = X[i]
        nans = np.isnan(row)
        if nans.any() and not nans.all():
            row[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), row[~nans])
            X[i] = row

    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True) + 1e-8
    X = (X - mu) / sd

    print(f"Loaded {X.shape[0]} individual cell trajectories "
          f"from {measure.shape[0]} replicate(s), {X.shape[1]} timepoints each "
          f"(downsample={downsample}).")
    return X, replicate_id, rep_name
