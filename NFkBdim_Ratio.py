import h5py
import numpy as np
import matplotlib.pyplot as plt

def decode_uint16_string(arr):
    arr = np.array(arr).flatten()
    return "".join(chr(int(x)) for x in arr if int(x) != 0)

with h5py.File("AllDataSets.mat", "r") as f:
    ds = f["DataSets"]

    data_refs = ds["Data"][()]
    repname_refs = ds["RepNames"][()]

    i = 0  # LPS_R1

    rep_name = decode_uint16_string(f[repname_refs[i, 0]][()])
    print("Condition:", rep_name)

    data_group = f[data_refs[i, 0]]
    measure = data_group["measure"][()]

    signals = []

    for j in range(measure.shape[0]):
        ref = measure[j, 0]
        obj = f[ref]

        # 🔥 Extract correct field
        nfkb = obj["NFkBdim_Ratio"][()]

        # shape = (time, cells)
        # flatten across cells
        nfkb = np.array(nfkb)

        # average across cells
        signal = np.nanmean(nfkb, axis=1)

        signals.append(signal)

    # stack
    signals = np.array(signals)

    print("Final signal matrix:", signals.shape)

    # 🔥 Downsample
    signals = signals[:, ::10]

    # 🔥 Normalize
    signals = (signals - signals.mean(axis=1, keepdims=True)) / signals.std(axis=1, keepdims=True)

    # Plot
    for s in signals:
        plt.plot(s, alpha=0.7)

    plt.title(f"NF-kB Signals (Ratio): {rep_name}")
    plt.xlabel("Time")
    plt.ylabel("Normalized Signal")
    plt.show()
