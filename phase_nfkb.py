import h5py
import numpy as np
import matplotlib.pyplot as plt

MAT_FILE = "AllDataSets.mat"
CONDITION_INDEX = 0   # LPS_R1
DOWNSAMPLE = 10
SIGNAL_KEY = "NFkBdim_Ratio"


def decode_uint16_string(arr):
    arr = np.array(arr).flatten()
    return "".join(chr(int(x)) for x in arr if int(x) != 0)


# -------------------------------
# Load NF-kB signals
# -------------------------------
with h5py.File(MAT_FILE, "r") as f:
    ds = f["DataSets"]

    data_refs = ds["Data"][()]
    repname_refs = ds["RepNames"][()]

    rep_name = decode_uint16_string(f[repname_refs[CONDITION_INDEX, 0]][()])
    print("Condition:", rep_name)

    data_group = f[data_refs[CONDITION_INDEX, 0]]
    measure = data_group["measure"][()]

    signals = []

    for j in range(measure.shape[0]):
        ref = measure[j, 0]
        obj = f[ref]

        nfkb = np.array(obj[SIGNAL_KEY][()])
        signal = np.nanmean(nfkb, axis=1)
        signals.append(signal)

signals = np.array(signals)

# Downsample + normalize
signals = signals[:, ::DOWNSAMPLE]
signals = (signals - signals.mean(axis=1, keepdims=True)) / (
    signals.std(axis=1, keepdims=True) + 1e-8
)

# Mean trajectory
signal = np.mean(signals, axis=0)

# -------------------------------
# Phase-response calculation
# -------------------------------
window = 2

responses = []
times = range(window, len(signal) - window)

for t in times:
    pre = signal[t - window:t]
    post = signal[t:t + window]

    response = np.mean(post) - np.mean(pre)
    responses.append(response)

responses = np.array(responses)
times = np.array(list(times))

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(8,4))
plt.plot(times, responses)
plt.axhline(0, linestyle='--')
plt.title(f"NF-kB Phase Response: {rep_name}")
plt.xlabel("Perturbation timing")
plt.ylabel("Response magnitude")
plt.show()

# -------------------------------
# Metrics
# -------------------------------
prc_variance = np.var(responses)

print("PRC variance:", prc_variance)
print("Non-flat response:", prc_variance > 1e-6)
