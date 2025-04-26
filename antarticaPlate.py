import os
import numpy as np
import xarray as xr
import glob
import pyvista as pv
from scipy.ndimage import zoom

# Parameters
paleodem_dir = './Scotese_Wright_2018_Maps_1-88_1degX1deg_PaleoDEMS_nc_v2/'
output_dir = './frames_pyvista'
os.makedirs(output_dir, exist_ok=True)
use_interactive = True
time_start = 0
max_age = 100
step = 1
fixed_vmin = -6000
fixed_vmax = 4000
zoom_factor = 4
z_exaggeration = 30
earth_radius = 6371000
frames_per_transition = 50
time_frames = 100

camera_path = [
    ((23586575.715407833, -43262495.6267938, -15439462.140339784), (0, 0, 0), (0, 0, 1)),
    ((-33210817.19258376, 32212019.661417406, -22856656.617927868), (0, 0, 0), (0, 0, 1)),
    ((841697.1492205107, 2544088.6342996275, -51570785.35152652), (0, 0, 0), (0, 0, 1))
]

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

# Generate globe surface
def generate_globe(elevation, lons, lats):
    elevation_zoomed = zoom(elevation, zoom_factor)
    lon_zoomed = np.linspace(lons.min(), lons.max(), elevation_zoomed.shape[1])
    lat_zoomed = np.linspace(lats.min(), lats.max(), elevation_zoomed.shape[0])
    Lon, Lat = np.meshgrid(lon_zoomed, lat_zoomed)
    topo = elevation_zoomed * z_exaggeration
    Lon_flat = Lon.flatten()
    Lat_flat = Lat.flatten()
    Z_flat = topo.flatten()
    lon_rad = np.deg2rad(Lon_flat)
    lat_rad = np.deg2rad(Lat_flat)
    X0 = np.cos(lat_rad) * np.cos(lon_rad)
    Y0 = np.cos(lat_rad) * np.sin(lon_rad)
    Z0 = np.sin(lat_rad)
    norms = np.sqrt(X0**2 + Y0**2 + Z0**2)
    X0 /= norms
    Y0 /= norms
    Z0 /= norms
    r = earth_radius + Z_flat
    X = r * X0
    Y = r * Y0
    Z = r * Z0
    points = np.column_stack((X, Y, Z))
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (Lon.shape[1], Lon.shape[0], 1)
    
    # IMPORTANT: color scalars are based on original elevation (not exaggerated topo)
    elevation_flat = elevation_zoomed.flatten()
    scalars = np.clip(elevation_flat, fixed_vmin, fixed_vmax)

    return grid, scalars

# Load DEM information
age_file_map = load_available_dems()
available_ages = list(age_file_map.keys())

if use_interactive:
    plotter = pv.Plotter(off_screen=False)
    plotter.enable_lightkit()

    def capture_camera():
        cam = plotter.camera
        print(f"Captured camera: {cam.position}, {cam.focal_point}, {cam.up}")

    plotter.add_key_event('c', capture_camera)

    frame_age = 0
    result = interpolate_dem(frame_age, available_ages, age_file_map)

    if result is not None:
        elevation, lons, lats = result
        grid, scalars = generate_globe(elevation, lons, lats)
        plotter.add_mesh(grid, scalars=scalars, cmap="terrain", clim=[fixed_vmin, fixed_vmax])
        plotter.set_background("black")
        plotter.camera.zoom(1.5)
        print("Interactive mode launched. Press 'c' to capture camera.")
        plotter.show()
    else:
        print(f"No DEM data available for {frame_age} Ma.")

else:
    plotter = pv.Plotter(off_screen=True)
    plotter.enable_lightkit()

    step_counter = 0

    for i in range(len(camera_path) - 1):
        pos0, focal0, up0 = camera_path[i]
        pos1, focal1, up1 = camera_path[i + 1]
        for t in np.linspace(0, 1, frames_per_transition):
            pos = np.array(pos0) * (1 - t) + np.array(pos1) * t
            focal = np.array(focal0) * (1 - t) + np.array(focal1) * t
            up = np.array(up0) * (1 - t) + np.array(up1) * t

            result = interpolate_dem(0, available_ages, age_file_map)
            if result is None:
                continue

            elevation, lons, lats = result
            plotter.clear()
            grid, scalars = generate_globe(elevation, lons, lats)
            plotter.add_mesh(grid, scalars=scalars, cmap="terrain", clim=[fixed_vmin, fixed_vmax])
            plotter.set_background("black")
            plotter.camera.position = pos
            plotter.camera.focal_point = focal
            plotter.camera.up = up
            plotter.show(screenshot=os.path.join(output_dir, f"path_frame_{step_counter:04d}.png"), auto_close=False, window_size=(1920, 1080))
            print(f"Saved transition frame {step_counter}")
            step_counter += 1

    last_pos, last_focal, last_up = camera_path[-1]
    for frame_age in range(time_start, max_age + step, step):
        result = interpolate_dem(frame_age, available_ages, age_file_map)
        if result is None:
            continue
        elevation, lons, lats = result
        plotter.clear()
        grid, scalars = generate_globe(elevation, lons, lats)
        plotter.add_mesh(grid, scalars=scalars, cmap="terrain", clim=[fixed_vmin, fixed_vmax])
        plotter.set_background("black")
        plotter.camera.position = last_pos
        plotter.camera.focal_point = last_focal
        plotter.camera.up = last_up
        plotter.show(screenshot=os.path.join(output_dir, f"time_frame_{step_counter:04d}.png"), auto_close=False, window_size=(1920, 1080))
        print(f"Saved time frame for {frame_age} Ma")
        step_counter += 1

    plotter.close()
