import h5py

def store_to_hdf5(result_dict, path):
    array_map = {
        "vv_db": "vv_db",
        "vh_db": "vh_db",
        "water_mask": "water_mask"
    }
    with h5py.File(path, "w") as f:
        group = f.create_group("Satellite Data")
        for key, ds_name in array_map.items():
            if key in result_dict and result_dict[key] is not None:
                group.create_dataset(
                    ds_name,
                    data=result_dict[key],
                    compression="gzip",
                    compression_opts=4
                )
                
def store_to_hdf5_s2(result_dict, path):
    array_map = {
        "ndwi": "ndwi",
        "ndvi": "ndvi",
        "ndmi": "ndmi",
        "rgb_u8": "rgb_u8"
    }
    with h5py.File(path, "w") as f:
        group = f.create_group("Satellite Data")
        for key, ds_name in array_map.items():
            if key in result_dict and result_dict[key] is not None:
                group.create_dataset(
                    ds_name,
                    data=result_dict[key],
                    compression="gzip",
                    compression_opts=4
                )   