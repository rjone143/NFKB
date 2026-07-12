import h5py
import numpy as np

file = h5py.File("AllDataSets.mat", "r")
ds = file["DataSets"]

for key in ["Data", "RepIDs", "RepNames"]:
    obj = ds[key]
    print(f"\n--- {key} ---")
    print("type:", type(obj))
    try:
        print("shape:", obj.shape)
    except Exception:
        pass
    try:
        print("dtype:", obj.dtype)
    except Exception:
        pass

    # Try to show a small preview for datasets
    if isinstance(obj, h5py.Dataset):
        try:
            arr = obj[()]
            print("preview:", arr[:5] if hasattr(arr, "__len__") else arr)
        except Exception as e:
            print("preview error:", e)

    # If it's a group, show keys
    if isinstance(obj, h5py.Group):
        print("keys:", list(obj.keys()))
