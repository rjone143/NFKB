import h5py
import numpy as np

file = h5py.File("AllDataSets.mat", "r")
repnames = file["DataSets"]["RepNames"]

print("RepNames type:", type(repnames))
print("RepNames shape:", getattr(repnames, "shape", None))
print("RepNames dtype:", getattr(repnames, "dtype", None))

if isinstance(repnames, h5py.Dataset):
    arr = repnames[()]
    print("Raw RepNames preview:", arr[:10])

    # If these are references, dereference the first few
    flat = arr.flatten()
    for i, ref in enumerate(flat[:5]):
        try:
            obj = file[ref]
            print(f"\nRepName {i}: type={type(obj)} shape={getattr(obj, 'shape', None)} dtype={getattr(obj, 'dtype', None)}")
            if isinstance(obj, h5py.Dataset):
                data = obj[()]
                print("raw:", data[:20] if hasattr(data, "__len__") else data)
        except Exception as e:
            print(f"RepName {i} dereference error:", e)
