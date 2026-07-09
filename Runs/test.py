import glob, os
import xarray as xr
import h5py

def read_climate_data(dir_path: str = os.path.join('..', 'Data', 'HydroGFD', 'data_files')) -> xr.Dataset:
    files = sorted(glob.glob(os.path.join(dir_path, '*.nc')))

    datasets = [
    xr.open_dataset(
        f,
        engine="h5netcdf",
        chunks={"time":365}
    )
    for f in files
    ]

    ds = xr.combine_by_coords(
        datasets,
        combine_attrs="override"
    )

    return ds
ds = read_climate_data()
print(ds)