import xarray as xr

ds = xr.load_dataset('ERA5_Reanalysis_SingleLevels_1960.grib', engine='cfgrib')

print(ds)