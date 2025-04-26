import os
import pygplates
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import xarray as xr

# Parameters
rotation_file = './Muller_etal_2019_PlateMotionModel_v2.0_Tectonics_Updated/Global_250-0Ma_Rotations_2019_v2.rot'
topology_file = './Muller_etal_2019_PlateMotionModel_v2.0_Tectonics_Updated/SimplifiedFiles/Muller_etal_2019_Global_Coastlines.gpmlz'
paleodem_dir = './Scotese_Wright_2018_Maps_1-88_1degX1deg_PaleoDEMS_nc_v2/'

time_start = 0  # present day
max_age = 100  # 100 million years ago
step = 1  # step in million years
projection_center = (0, -90)  # Center on South Pole

# Load rotation model and topology
rotation_model = pygplates.RotationModel(rotation_file)
topology_features = pygplates.FeatureCollection(topology_file)

# Prepare figure
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.Orthographic(central_longitude=projection_center[0], central_latitude=projection_center[1]))
ax.set_global()
ax.gridlines(draw_labels=True)

# Function to find the closest DEM file for a given age
def find_closest_dem(age):
    available_ages = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    closest_age = min(available_ages, key=lambda x: abs(x - age))
    filename = f'Map{str(closest_age//5*2+1).zfill(2)}_PALEOMAP_1deg_*.nc'
    return closest_age, filename

# Function to update each frame
def update(frame_age):
    ax.clear()
    ax.set_global()
    ax.gridlines(draw_labels=True)
    ax.set_title(f"Antarctica centered - {frame_age} Ma", fontsize=12)

    # Find the closest DEM
    dem_age, _ = find_closest_dem(frame_age)
    dem_path = os.path.join(paleodem_dir, f'Map{str(dem_age//5*2+1).zfill(2)}_PALEOMAP_1deg_*.nc')

    # Find the actual file (wildcard)
    import glob
    dem_files = glob.glob(dem_path)
    if dem_files:
        dem_file = dem_files[0]
        dem_data = xr.open_dataset(dem_file)
        var_name = list(dem_data.data_vars.keys())[0]
        topo = dem_data[var_name]

        lons = topo.coords['lon'].values
        lats = topo.coords['lat'].values
        elevation = topo.values

        Lon, Lat = np.meshgrid(lons, lats)

        ax.pcolormesh(Lon, Lat, elevation, cmap='terrain', shading='auto', transform=ccrs.PlateCarree())

    reconstructed_features = []
    pygplates.reconstruct(topology_features, rotation_model, reconstructed_features, frame_age)

    for feature in reconstructed_features:
        geometry = feature.get_reconstructed_geometry()
        if geometry:
            lon_lat_points = geometry.to_lat_lon_array()
            lons, lats = lon_lat_points[:, 1], lon_lat_points[:, 0]
            ax.plot(lons, lats, transform=ccrs.Geodetic(), color='black', linewidth=0.5)

# Create frames
ages = list(range(time_start, max_age + step, step)) + list(range(max_age, time_start - step, -step))
ani = animation.FuncAnimation(fig, update, frames=ages, interval=200, repeat=True)

plt.show()