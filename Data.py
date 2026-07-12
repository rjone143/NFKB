import h5py

file = h5py.File("AllDataSets.mat", "r")
data = file["DataSets"]["Data"]

print("Data type:", type(data))
print("Data shape:", getattr(data, "shape", None))
print("Data dtype:", getattr(data, "dtype", None))

if isinstance(data, h5py.Dataset):
    arr = data[()]
    print("Raw Data preview type:", type(arr))
    print("Raw Data preview shape:", getattr(arr, "shape", None))
    print("First few entries:", arr.flatten()[:10])
elif isinstance(data, h5py.Group):
    print("Data keys:", list(data.keys()))
