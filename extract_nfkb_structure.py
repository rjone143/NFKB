import h5py
import numpy as np

def decode_uint16_string(arr):
    arr = np.array(arr).flatten()
    return "".join(chr(int(x)) for x in arr if int(x) != 0)

with h5py.File("AllDataSets.mat", "r") as f:
    ds = f["DataSets"]

    data_refs = ds["Data"][()]
    repid_refs = ds["RepIDs"][()]
    repname_refs = ds["RepNames"][()]

    print("Decoded entries:\n")

    for i in range(data_refs.shape[0]):
        rep_name = decode_uint16_string(f[repname_refs[i, 0]][()])
        repid_obj = f[repid_refs[i, 0]]
        data_obj = f[data_refs[i, 0]]

        print(f"Index {i}")
        print(f"  RepName: {rep_name}")
        print(f"  Data object type: {type(data_obj)}")
        print(f"  RepIDs object type: {type(repid_obj)}")

        if isinstance(data_obj, h5py.Group):
            print(f"  Data group keys: {list(data_obj.keys())}")
            for k in data_obj.keys():
                sub = data_obj[k]
                print(f"    - {k}: type={type(sub)}", end="")
                if isinstance(sub, h5py.Dataset):
                    print(f", shape={sub.shape}, dtype={sub.dtype}")
                else:
                    print()
        elif isinstance(data_obj, h5py.Dataset):
            print(f"  Data dataset shape: {data_obj.shape}, dtype={data_obj.dtype}")

        if isinstance(repid_obj, h5py.Dataset):
            try:
                repid_arr = repid_obj[()]
                print(f"  RepIDs shape: {repid_arr.shape}, dtype={repid_arr.dtype}")
                print(f"  RepIDs preview: {np.array(repid_arr).flatten()[:10]}")
            except Exception as e:
                print(f"  RepIDs read error: {e}")
        elif isinstance(repid_obj, h5py.Group):
            print(f"  RepIDs group keys: {list(repid_obj.keys())}")

        print()
