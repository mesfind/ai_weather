import xarray as xr
import xesmf as xe
import glob
import os

# ------------------------------
# Paths
data_dir = "./"                 # folder with rr_mrg_*.nc
target_grid_file = "AIFS_2019.nc"
output_dir = "./regridded"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# Load target grid
ds_grid = xr.open_dataset(target_grid_file)
# Assumes ds_grid has lat/lon coordinates
# You can also rename coords if needed: ds_grid = ds_grid.rename({"Lat":"lat","Lon":"lon"})

# ------------------------------
# List of yearly files
files = sorted(glob.glob(os.path.join(data_dir, "rr_mrg_20*.nc")))

# ------------------------------
# Loop through each file
for f in files:
    print(f"Processing {f} ...")
    
    # Open the dataset
    ds = xr.open_dataset(f)
    
    # Ensure lat/lon names match the target grid
    # Adjust if necessary
    if 'Lat' in ds.coords:
        ds = ds.rename({"Lat": "lat"})
    if 'Lon' in ds.coords:
        ds = ds.rename({"Lon": "lon"})
    
    # Create regridder
    regridder = xe.Regridder(ds, ds_grid, method='bilinear')#, reuse_weights=True)
    
    # Apply regridding to 'precip'
    ds_out = regridder(ds['precip'])
    
    # Convert back to a Dataset (optional)
    ds_out = ds_out.to_dataset(name='precip')
    
    # Add coordinates
    ds_out = ds_out.assign_coords({'lat': ds_grid['lat'], 'lon': ds_grid['lon']})
    
    # Save to new NetCDF
    #year = os.path.basename(f).split('_')[2]  # extract year from filename
    year = os.path.splitext(os.path.basename(f))[0].split('_')[2]
    out_file = os.path.join(output_dir, f"rr_mrg_{year}_025.nc")
    ds_out.to_netcdf(out_file)
    
    print(f"Saved regridded file: {out_file}")

print("All files regridded successfully!")
