import h5py
import numpy as np
import matplotlib.pyplot as plt

def decode_uint16_string(arr):
    arr = np.array(arr).flatten()
    return "".join(chr(int(x)) for x in arr if int(x) != 0)

def unwrap_ref(x):
    """Recursively unwrap MATLAB/HDF5 object-ref containers."""
    while isinstance(x, (tuple, list, np.ndarray)):
        x = x[0]
    return x

with h5py.File("AllDataSets.mat", "r") as f:
    ds = f["DataSets"]

    data_refs = ds["Data"][()]
    repname_refs = ds["RepNames"][()]

    i = 0  # LPS_R1
    rep_name = decode_uint16_string(f[repname_refs[i, 0]][()])
    print("Condition:", rep_name)

    data_group = f[data_refs[i, 0]]
    measure = data_group["measure"][()]

    print("measure shape:", measure.shape)
    print("sample entry type:", type(measure[0, 0]))
    print("sample entry repr:", measure[0, 0])

    signals = []

    for j in range(measure.shape[0]):
        ref = unwrap_ref(measure[j, 0])
        obj = f[ref]

        print(f"\nEntry {j}: type={type(obj)}")

        if isinstance(obj, h5py.Dataset):
            arr = np.array(obj[()]).flatten()
            print("  dataset shape:", obj.shape, "dtype:", obj.dtype)
            print("  preview:", arr[:10])
            signals.append(arr)

        elif isinstance(obj, h5py.Group):
            print("  group keys:", list(obj.keys()))
            # try to find the actual numeric dataset inside
            found = False
            for k in obj.keys():
                sub = obj[k]
                print(f"   - {k}: type={type(sub)}", end="")
                if isinstance(sub, h5py.Dataset):
                    print(f", shape={sub.shape}, dtype={sub.dtype}")
                    arr = np.array(sub[()]).flatten()
                    print("     preview:", arr[:10])
                    # keep first numeric-looking dataset
                    if np.issubdtype(arr.dtype, np.number):
                        signals.append(arr)
                        found = True
                        break
                else:
                    print()
            if not found:
                print("  no numeric dataset extracted from this entry")

    if signals:
        min_len = min(len(s) for s in signals)
        signals = np.array([s[:min_len] for s in signals])
        print("\nSignal matrix shape:", signals.shape)

        for s in signals:
            plt.plot(s, alpha=0.7)

        plt.title(f"NF-kB Signals: {rep_name}")
        plt.xlabel("Time")
        plt.ylabel("Signal")
        plt.show()
    else:
        print("\nNo signals extracted yet.")
