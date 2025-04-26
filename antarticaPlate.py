import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import xarray as xr
import glob
from scipy.ndimage import zoom
from matplotlib.colors import LightSource

# Parameters
paleodem_dir = './Scotese_Wright_2018_Maps_1-88_1degX1deg_PaleoDEMS_nc_v2/'
time_start = 0  # present day
max_age = 100  # 100 million years ago
step = 1  # 1 million year step
projection_center = (0, -90)  # Center on South Pole
fixed_vmin = -6000
fixed_vmax = 4000
zoom_factor = 4  # Spatial upscaling factor
lightsource_azdeg = 315  # Azimuth of the light source
lightsource_altdeg = 45  # Altitude of the light source

# Load available DEMs
def load_available_dems():
    dem_files = glob.glob(os.path.join(paleodem_dir, 'Map*_PALEOMAP_1deg_*.nc'))
    age_file_map = {}
    for dem in dem_files:
        filename = os.path.basename(dem)
        parts = filename.split('_')
        for p in parts:
            if 'Ma' in p:
                age_str = p.replace('Ma.nc', '').replace('Ma', '')
                try:
                    age = float(age_str)
                    age_file_map[age] = dem
                    print(f"Loaded DEM for {age} Ma: {filename}")
                except ValueError:
                    pass
    return dict(sorted(age_file_map.items()))

# Interpolate DEM between two frames
def interpolate_dem(age, available_ages, age_file_map):
    lower_ages = [a for a in available_ages if a <= age]
    upper_ages = [a for a in available_ages if a >= age]

    if not lower_ages or not upper_ages:
        return None

    age_lower = max(lower_ages)
    age_upper = min(upper_ages)

    if age_lower == age_upper:
        dem_data = xr.open_dataset(age_file_map[age_lower])
        var_name = list(dem_data.data_vars.keys())[0]
        topo = dem_data[var_name]
        lons = topo.coords['lon'].values
        lats = topo.coords['lat'].values
        elevation = topo.values
        return elevation, lons, lats

    weight_upper = (age - age_lower) / (age_upper - age_lower)
    weight_lower = 1.0 - weight_upper

    dem_data_lower = xr.open_dataset(age_file_map[age_lower])
    dem_data_upper = xr.open_dataset(age_file_map[age_upper])

    var_name_lower = list(dem_data_lower.data_vars.keys())[0]
    var_name_upper = list(dem_data_upper.data_vars.keys())[0]

    data_lower = dem_data_lower[var_name_lower].values
    data_upper = dem_data_upper[var_name_upper].values

    interpolated = weight_lower * data_lower + weight_upper * data_upper

    lons = dem_data_lower['lon'].values
    lats = dem_data_lower['lat'].values

    return interpolated, lons, lats

# Prepare figure
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.Orthographic(central_longitude=projection_center[0], central_latitude=projection_center[1]))
ax.set_global()
ax.gridlines(draw_labels=True)

# Load DEM info
age_file_map = load_available_dems()
available_ages = list(age_file_map.keys())
ls = LightSource(azdeg=lightsource_azdeg, altdeg=lightsource_altdeg)

# Function to update each frame
def update(frame_age):
    ax.clear()
    ax.set_global()
    ax.gridlines(draw_labels=True)
    ax.set_title(f"Antarctica centered - {frame_age} Ma", fontsize=12)

    result = interpolate_dem(frame_age, available_ages, age_file_map)
    if result is None:
        return

    elevation, lons, lats = result

    # Upscale for smoothness
    elevation_zoomed = zoom(elevation, zoom_factor)
    lon_zoomed = np.linspace(lons.min(), lons.max(), elevation_zoomed.shape[1])
    lat_zoomed = np.linspace(lats.min(), lats.max(), elevation_zoomed.shape[0])
    Lon, Lat = np.meshgrid(lon_zoomed, lat_zoomed)

    # Apply light shading and create RGB image
    rgb = ls.shade(elevation_zoomed, cmap=plt.cm.terrain, vert_exag=0.1, blend_mode='soft', vmin=fixed_vmin, vmax=fixed_vmax)

    # Display RGB image
    ax.imshow(rgb, origin='lower', extent=[lon_zoomed.min(), lon_zoomed.max(), lat_zoomed.min(), lat_zoomed.max()], transform=ccrs.PlateCarree())

# Create frames
ages = list(range(time_start, max_age + step, step))
ani = animation.FuncAnimation(fig, update, frames=ages, interval=300, repeat=True)

plt.show()