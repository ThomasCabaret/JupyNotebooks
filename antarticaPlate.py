import os
import time
import numpy as np
import xarray as xr
import glob
import pyvista as pv
from scipy.ndimage import zoom

output_dir = './frames_pyvista_antartica'
os.makedirs(output_dir, exist_ok=True)

# Parameters
paleodem_dir = './Scotese_Wright_2018_Maps_1-88_1degX1deg_PaleoDEMS_nc_v2/'
demo_mode = True  # <----- SET THIS TO False IF YOU WANT MANUAL INTERACTION

# Other settings
fixed_vmin = -6000
fixed_vmax = 4000
zoom_factor = 4
z_exaggeration = 30
earth_radius = 6371000
frame_rate = 25  # frames per second
dt = 1.0 / frame_rate

# Camera path
camera_path = [
    ((21670351.848707743, -35169177.58583997, -10724692.858335884),
     (0, 0, 0),
     (0.31731259647336785, -0.0916657434823759, 0.9438803460138119)),
    ((-28280247.29284047, 26313815.240560796, -18070005.422952995),
     (0, 0, 0),
     (-0.41950473259149795, 0.16117930701558358, 0.8933291724349497)),
    ((-2747502.2130420175, -572113.80069213, -42590227.82606688),
     (0, 0, 0),
     (0.7960342683664514, 0.6023034913707767, -0.059665298699170796))
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

    elevation_flat = elevation_zoomed.flatten()
    scalars = np.clip(elevation_flat, fixed_vmin, fixed_vmax)
    return grid, scalars

# Load DEM information
age_file_map = load_available_dems()
available_ages = list(age_file_map.keys())

# Setup plotter
plotter = pv.Plotter(off_screen=False, window_size=(1920, 1080))
plotter.enable_lightkit()

frame_age = 0
frame_counter = 0
result = interpolate_dem(frame_age, available_ages, age_file_map)
if result is None:
    raise RuntimeError(f"No DEM data available for {frame_age} Ma.")
elevation, lons, lats = result
grid, scalars = generate_globe(elevation, lons, lats)
mesh_actor = plotter.add_mesh(grid, scalars=scalars, cmap="terrain", clim=[fixed_vmin, fixed_vmax], show_scalar_bar=False)
plotter.set_background("black")
plotter.camera.zoom(1.5)

if demo_mode:
    # DEMO MODE
    plotter.show(interactive_update=True, auto_close=False)

    for i in range(len(camera_path) - 1):
        pos0, focal0, up0 = camera_path[i]
        pos1, focal1, up1 = camera_path[i + 1]
        for t in np.linspace(0, 1, 50):
            pos = np.array(pos0) * (1 - t) + np.array(pos1) * t
            focal = np.array(focal0) * (1 - t) + np.array(focal1) * t
            up = np.array(up0) * (1 - t) + np.array(up1) * t

            plotter.camera.position = pos
            plotter.camera.focal_point = focal
            plotter.camera.up = up
            plotter.update()
            filename = os.path.join(output_dir, f"frame_{frame_counter:04d}.png")
            plotter.screenshot(filename)
            frame_counter += 1
            time.sleep(dt)

    last_pos, last_focal, last_up = camera_path[-1]
    plotter.camera.position = last_pos
    plotter.camera.focal_point = last_focal
    plotter.camera.up = last_up

    # Temporal transition (remonter le temps)
    for frame_age in np.linspace(0, 100, 100):
        result = interpolate_dem(frame_age, available_ages, age_file_map)
        if result is None:
            continue
        elevation, lons, lats = result
        grid, scalars = generate_globe(elevation, lons, lats)
        plotter.remove_actor(mesh_actor)
        mesh_actor = plotter.add_mesh(grid, scalars=scalars, cmap="terrain", clim=[fixed_vmin, fixed_vmax], show_scalar_bar=False)
        plotter.update()
        filename = os.path.join(output_dir, f"frame_{frame_counter:04d}.png")
        plotter.screenshot(filename)
        frame_counter += 1
        time.sleep(dt)

    plotter.close()
else:
    # INTERACTIVE MODE
    def capture_camera():
        cam = plotter.camera
        print(f"Captured camera:\n  Position: {cam.position}\n  Focal Point: {cam.focal_point}\n  Up Vector: {cam.up}")

    plotter.add_key_event('c', capture_camera)
    print("Interactive mode active. Press 'c' to capture camera.")
    plotter.show()
