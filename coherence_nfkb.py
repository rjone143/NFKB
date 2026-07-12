import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

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

        nfkb = obj["NFkBdim_Ratio"][()]
        nfkb = np.array(nfkb)

        signal = np.nanmean(nfkb, axis=1)
        signals.append(signal)

    signals = np.array(signals)

# 🔥 Downsample
signals = signals[:, ::10]

# 🔥 Normalize
signals = (signals - signals.mean(axis=1, keepdims=True)) / signals.std(axis=1, keepdims=True)

# 🔥 Transpose → time x cells
X = signals.T

# -------------------------------
# Coherence calculation
# -------------------------------
cross_var = np.var(X, axis=1)
coherence = 1 / (cross_var + 1e-6)
coherence = gaussian_filter1d(coherence, sigma=2)

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(8,4))
plt.plot(coherence)
plt.title(f"NF-kB Coherence: {rep_name}")
plt.xlabel("Time")
plt.ylabel("1 / variance")
plt.show()

# -------------------------------
# Debug output
# -------------------------------
peak_idx = np.argmax(coherence)
print("Peak coherence index:", peak_idx)
