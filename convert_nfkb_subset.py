import numpy as np
import pandas as pd
from scipy.io import loadmat

print("Loading data...")

# Load with variable filtering
data = loadmat("AllDataSets.mat", simplify_cells=True)

print("Available keys:")
print(data.keys())

# 🔴 You MUST adjust this after seeing keys
# Typical structure guess:
dataset = data["AllDataSets"]

# Inspect structure
print(type(dataset))

# Try accessing first condition
sample = dataset[0]
print(sample.keys())

# Example assumptions (you may tweak based on structure)
# Look for something like:
# sample["measure"]["NFkB"] or similar

nfkb = sample["measure"]["NFkB"]

print("NFkB shape:", np.shape(nfkb))

# Take only first 50 cells (to keep it lightweight)
nfkb = nfkb[:50]

rows = []

for cell_id, signal in enumerate(nfkb):
    for t, value in enumerate(signal):
        rows.append({
            "cell_id": cell_id,
            "time": t,
            "signal": float(value),
            "outcome": "high" if np.max(signal) > 0.5 else "low"
        })

df = pd.DataFrame(rows)

df.to_csv("nfkb_timeseries.csv", index=False)

print("Saved nfkb_timeseries.csv")
