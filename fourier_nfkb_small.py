import h5py
import numpy as np
import matplotlib.pyplot as plt

MAT_FILE = "AllDataSets.mat"
CONDITION_INDEX = 0   # 0=LPS_R1
SIGNAL_KEY = "NFkBdim_Ratio"
DOWNSAMPLE = 2        # much lighter than 10; keeps more temporal structure


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

# keep more time points than before
signals = signals[:, ::DOWNSAMPLE]

# z-score each replicate
signals = (signals - signals.mean(axis=1, keepdims=True)) / (
    signals.std(axis=1, keepdims=True) + 1e-8
)

# use mean trajectory for a simple local test
mean_signal = np.mean(signals, axis=0)
n = len(mean_signal)

# FFT
fft_vals = np.fft.rfft(mean_signal)
power = np.abs(fft_vals) ** 2
freqs = np.fft.rfftfreq(n, d=1.0)  # unit spacing; relative frequency only

# remove DC for dominant-frequency lookup
power_no_dc = power.copy()
if len(power_no_dc) > 0:
    power_no_dc[0] = 0

dominant_idx = np.argmax(power_no_dc)
dominant_freq = freqs[dominant_idx]
dominant_power = power[dominant_idx]

# plots
plt.figure(figsize=(8, 4))
plt.plot(mean_signal)
plt.title(f"Mean NF-kB Signal: {rep_name}")
plt.xlabel("Time")
plt.ylabel("Normalized signal")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(freqs, power, marker="o")
plt.title(f"FFT Power Spectrum: {rep_name}")
plt.xlabel("Relative frequency")
plt.ylabel("Power")
plt.tight_layout()
plt.show()

print("Number of time points:", n)
print("Dominant frequency index:", dominant_idx)
print("Dominant relative frequency:", float(dominant_freq))
print("Dominant power:", float(dominant_power))
print("Top 5 power values:", np.sort(power)[-5:])
