import h5py
import numpy as np
import matplotlib.pyplot as plt

MAT_FILE = "AllDataSets.mat"
CONDITION_INDEX = 0   # 0=LPS_R1
DOWNSAMPLE = 10
SIGNAL_KEY = "NFkBdim_Ratio"


def decode_uint16_string(arr):
    arr = np.array(arr).flatten()
    return "".join(chr(int(x)) for x in arr if int(x) != 0)


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

        nfkb = np.array(obj[SIGNAL_KEY][()])   # shape usually time x cells
        signal = np.nanmean(nfkb, axis=1)      # average across cells
        signals.append(signal)

signals = np.array(signals)

# Downsample and normalize per replicate
signals = signals[:, ::DOWNSAMPLE]
signals = (signals - signals.mean(axis=1, keepdims=True)) / (
    signals.std(axis=1, keepdims=True) + 1e-8
)

# Average replicate trajectory
mean_signal = np.nanmean(signals, axis=0)

# Binary activated/inactive state
threshold = np.nanmean(mean_signal)
state = (mean_signal > threshold).astype(int)

# Extract dwell / residence times
res_times = []
count = 1

for i in range(1, len(state)):
    if state[i] == state[i - 1]:
        count += 1
    else:
        res_times.append(count)
        count = 1

res_times.append(count)
res_times = np.array(res_times)

# Markov null with same mean dwell time
p = 1.0 / np.mean(res_times)
markov_times = np.random.geometric(p=min(max(p, 1e-6), 0.999), size=len(res_times))

# Tail metric
tail_threshold = np.percentile(res_times, 75)
tail_ratio_data = np.mean(res_times >= tail_threshold)
tail_ratio_markov = np.mean(markov_times >= tail_threshold)

# Plot 1: mean signal and threshold
plt.figure(figsize=(8, 4))
plt.plot(mean_signal, label="Mean NF-kB signal")
plt.axhline(threshold, color="red", linestyle="--", label="Threshold")
plt.title(f"NF-kB Mean Signal and State Threshold: {rep_name}")
plt.xlabel("Time")
plt.ylabel("Normalized signal")
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: binary state
plt.figure(figsize=(8, 2.5))
plt.step(np.arange(len(state)), state, where="mid")
plt.title(f"NF-kB Binary State: {rep_name}")
plt.xlabel("Time")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# Plot 3: dwell distribution
bins = np.logspace(np.log10(1), np.log10(max(res_times.max(), markov_times.max())), 20)

hist_data, edges = np.histogram(res_times, bins=bins, density=True)
hist_markov, _ = np.histogram(markov_times, bins=bins, density=True)
centers = np.sqrt(edges[:-1] * edges[1:])

plt.figure(figsize=(8, 4))
plt.plot(centers, hist_data, "o-", label="NF-kB")
plt.plot(centers, hist_markov, "o--", label="Markov null")
plt.xscale("log")
plt.yscale("log")
plt.title(f"Residence-Time Distribution: {rep_name}")
plt.xlabel("Residence time")
plt.ylabel("Probability density")
plt.legend()
plt.tight_layout()
plt.show()

print("Mean residence time:", float(np.mean(res_times)))
print("Mean Markov time:", float(np.mean(markov_times)))
print("Tail threshold:", float(tail_threshold))
print("Tail ratio (NF-kB):", float(tail_ratio_data))
print("Tail ratio (Markov):", float(tail_ratio_markov))
print("Residence times:", res_times)
