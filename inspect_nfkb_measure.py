import h5py
import numpy as np
import matplotlib.pyplot as plt


MAT_FILE = "AllDataSets.mat"
CONDITION_INDEX = 0       # 0=LPS_R1, 1=LPS_R2, 2=TNF_R1, ...
DOWNSAMPLE = 100          # increase/decrease if needed
MAX_PREVIEW_VALUES = 10


def decode_uint16_string(arr: np.ndarray) -> str:
    arr = np.array(arr).flatten()
    return "".join(chr(int(x)) for x in arr if int(x) != 0)


def unwrap_ref(x):
    while isinstance(x, (tuple, list, np.ndarray)):
        x = x[0]
    return x


def summarize_array(arr: np.ndarray) -> str:
    arr = np.array(arr)
    flat = arr.flatten()

    if flat.size == 0:
        return "empty"

    if np.issubdtype(flat.dtype, np.number):
        finite = flat[np.isfinite(flat)] if np.issubdtype(flat.dtype, np.floating) else flat
        if finite.size == 0:
            return f"shape={arr.shape}, dtype={arr.dtype}, all non-finite"
        return (
            f"shape={arr.shape}, dtype={arr.dtype}, "
            f"min={np.min(finite):.4f}, max={np.max(finite):.4f}, "
            f"mean={np.mean(finite):.4f}"
        )

    return f"shape={arr.shape}, dtype={arr.dtype}"


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    std = np.std(x)
    if std < 1e-12:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def score_candidate(arr: np.ndarray) -> float:
    """
    Heuristic:
    Prefer numeric 1D/flattenable arrays with enough length and variation.
    Penalize very large absolute scales a bit.
    """
    arr = np.array(arr).flatten()

    if arr.size < 100:
        return -1e9
    if not np.issubdtype(arr.dtype, np.number):
        return -1e9

    std = float(np.std(arr))
    if std < 1e-12:
        return -1e9

    length_score = np.log10(arr.size + 1)
    variation_score = np.log10(std + 1e-9)
    scale_penalty = np.log10(np.abs(np.mean(arr)) + 10.0)

    return 3.0 * length_score + 2.0 * variation_score - 0.5 * scale_penalty


def get_condition_name(f: h5py.File, repname_refs: np.ndarray, idx: int) -> str:
    return decode_uint16_string(f[repname_refs[idx, 0]][()])


def inspect_measure_entries(f: h5py.File, condition_index: int):
    ds = f["DataSets"]
    data_refs = ds["Data"][()]
    repname_refs = ds["RepNames"][()]

    rep_name = get_condition_name(f, repname_refs, condition_index)
    data_group = f[data_refs[condition_index, 0]]
    measure_refs = data_group["measure"][()]

    print(f"\nCondition: {rep_name}")
    print(f"measure shape: {measure_refs.shape}\n")

    all_candidates = []

    for j in range(measure_refs.shape[0]):
        print(f"===== measure entry {j} =====")
        ref = unwrap_ref(measure_refs[j, 0])
        obj = f[ref]

        print("Object type:", type(obj))

        if isinstance(obj, h5py.Dataset):
            arr = obj[()]
            print("Direct dataset:", summarize_array(arr))
            if np.issubdtype(np.array(arr).dtype, np.number):
                all_candidates.append((f"entry{j}::DIRECT", np.array(arr).flatten()))
            print()
            continue

        if isinstance(obj, h5py.Group):
            keys = list(obj.keys())
            print("Group keys:", keys)

            for k in keys:
                sub = obj[k]
                print(f"  - {k}: ", end="")

                if isinstance(sub, h5py.Dataset):
                    arr = np.array(sub[()])
                    print(summarize_array(arr))

                    flat = arr.flatten()
                    if np.issubdtype(flat.dtype, np.number):
                        score = score_candidate(flat)
                        all_candidates.append((f"entry{j}::{k}", flat))
                        preview = flat[:MAX_PREVIEW_VALUES]
                        print(f"      preview: {preview}")
                        print(f"      score: {score:.4f}")
                else:
                    print(f"type={type(sub)} (nested group)")

            print()

    return rep_name, all_candidates


def plot_best_candidates(rep_name: str, candidates: list[tuple[str, np.ndarray]], top_n: int = 6):
    if not candidates:
        print("No numeric candidates found.")
        return

    scored = []
    for name, arr in candidates:
        try:
            scored.append((score_candidate(arr), name, arr))
        except Exception:
            pass

    scored.sort(reverse=True, key=lambda x: x[0])

    print("\nTop candidate signals:")
    for score, name, arr in scored[:top_n]:
        print(f"  {name} | len={len(arr)} | score={score:.4f}")

    plt.figure(figsize=(12, 7))

    for score, name, arr in scored[:top_n]:
        y = arr[::DOWNSAMPLE]
        y = zscore_1d(y)
        plt.plot(y, alpha=0.8, label=name)

    plt.title(f"NF-kB Candidate Signals: {rep_name}")
    plt.xlabel(f"Time (downsampled by {DOWNSAMPLE})")
    plt.ylabel("Z-scored signal")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def main():
    with h5py.File(MAT_FILE, "r") as f:
        rep_name, candidates = inspect_measure_entries(f, CONDITION_INDEX)
        plot_best_candidates(rep_name, candidates, top_n=6)


if __name__ == "__main__":
    main()
