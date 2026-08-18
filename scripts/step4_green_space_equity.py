import osmnx as ox
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

BASE_DIR   = os.path.expanduser("~/Desktop/UrbanHeatHyderabad")
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

UHI_CLASS_PATH = os.path.join(OUTPUT_DIR, "uhi_classified_current.tif")
POP_PATH       = os.path.join(DATA_DIR, "population_hyderabad_2020.tif")

PRIORITY_UHI_CLASSES = [4, 5]
GREEN_SPACE_DISTANCE_THRESHOLD_KM = 1.0

print("Loading population raster (reference grid for equity analysis)...")
if not os.path.exists(POP_PATH):
    raise FileNotFoundError("Could not find: " + POP_PATH)

with rasterio.open(POP_PATH) as src:
    population = src.read(1).astype(np.float32)
    pop_profile = src.profile
    pop_transform = src.transform
    pop_crs = src.crs
    pop_shape = (src.height, src.width)
    pop_nodata = src.nodata

print("Population raster shape: " + str(pop_shape))
print("Pixel size: " + str(round(pop_transform.a, 1)) + " m")

if pop_nodata is not None:
    population = np.where(population == pop_nodata, 0, population)
population = np.where(population < 0, 0, population)

total_population = np.nansum(population)
print("Total population in study area: " + str(int(total_population)))

print("")
print("Loading UHI classified raster...")
with rasterio.open(UHI_CLASS_PATH) as src:
    uhi_class = src.read(1).astype(np.float32)
    uhi_transform = src.transform
    uhi_crs = src.crs

print("Resampling UHI classes onto population grid (nearest neighbor - categorical data)...")
uhi_on_pop_grid = np.empty(pop_shape, dtype=np.float32)
reproject(
    source=uhi_class,
    destination=uhi_on_pop_grid,
    src_transform=uhi_transform,
    src_crs=uhi_crs,
    dst_transform=pop_transform,
    dst_crs=pop_crs,
    resampling=Resampling.nearest
)
print("Resampling complete.")

# --- Fetch parks and green space from OpenStreetMap -------------------------
# IMPORTANT: osmnx expects bbox as (west, south, east, north) - i.e.
# (left, bottom, right, top). Hyderabad's actual extent is roughly
# 78.20 to 78.75 degrees East, 17.15 to 17.65 degrees North.
print("")
print("Fetching parks and green space from OpenStreetMap...")
print("(Querying a ~55km x 50km area - should take under a couple of minutes)")

west, south, east, north = 78.20, 17.15, 78.75, 17.65
print("Bounding box: west=" + str(west) + " south=" + str(south) + " east=" + str(east) + " north=" + str(north))

tags = {
    "leisure": ["park", "garden"],
    "landuse": ["recreation_ground", "forest"],
    "natural": ["wood"]
}

parks_gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags=tags)
print("Parks/green space features found: " + str(len(parks_gdf)))

# Keep only polygon geometries (parks are areas, not points/lines)
parks_gdf = parks_gdf[parks_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
print("Polygon park features: " + str(len(parks_gdf)))

# Reproject to match raster CRS
parks_gdf = parks_gdf.to_crs(pop_crs)

# --- Rasterize parks onto population grid ------------------------------------
print("")
print("Rasterizing park polygons onto population grid...")
park_shapes = [(geom, 1) for geom in parks_gdf.geometry if geom is not None and geom.is_valid]

park_raster = rasterize(
    shapes=park_shapes,
    out_shape=pop_shape,
    transform=pop_transform,
    fill=0,
    dtype="uint8"
)
park_pixel_count = np.sum(park_raster == 1)
print("Park pixels: " + str(park_pixel_count))

if park_pixel_count == 0:
    print("WARNING: No park pixels found. Check OSM query results or bounding box.")

# --- Distance to nearest park --------------------------------------------------
print("")
print("Computing distance to nearest park (Euclidean distance transform)...")
not_park = np.where(park_raster == 1, 0, 1)
distance_pixels = distance_transform_edt(not_park)
pixel_size_m = abs(pop_transform.a)
distance_km = (distance_pixels * pixel_size_m) / 1000.0

print("Distance range: " + str(round(np.nanmin(distance_km), 2)) + " to " + str(round(np.nanmax(distance_km), 2)) + " km")
print("Mean distance to nearest park: " + str(round(np.nanmean(distance_km), 2)) + " km")

# --- Identify priority intervention zones ------------------------------------
print("")
print("Identifying priority intervention zones...")
print("Criteria: UHI class High or Very High (4 or 5) AND distance to park > " + str(GREEN_SPACE_DISTANCE_THRESHOLD_KM) + " km")

high_heat_mask = np.isin(uhi_on_pop_grid, PRIORITY_UHI_CLASSES)
far_from_park_mask = distance_km > GREEN_SPACE_DISTANCE_THRESHOLD_KM
priority_zone_mask = high_heat_mask & far_from_park_mask

priority_pixel_count = np.sum(priority_zone_mask)
pixel_area_km2 = (pixel_size_m ** 2) / 1_000_000
priority_area_km2 = priority_pixel_count * pixel_area_km2

total_valid_pixels = np.sum(~np.isnan(uhi_on_pop_grid))
priority_pct_of_area = (priority_pixel_count / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0

print("Priority zone pixels: " + str(priority_pixel_count))
print("Priority zone area: " + str(round(priority_area_km2, 2)) + " sq km")
print("Priority zone as percent of study area: " + str(round(priority_pct_of_area, 1)) + " percent")

# --- Population living in priority zones --------------------------------------
print("")
print("Estimating population living in priority intervention zones...")
population_in_priority = np.nansum(np.where(priority_zone_mask, population, 0))
pct_population_at_risk = (population_in_priority / total_population) * 100 if total_population > 0 else 0

print("Population in priority zones: " + str(int(population_in_priority)))
print("Percent of total population: " + str(round(pct_population_at_risk, 1)) + " percent")

# --- Population by heat class (regardless of green space) --------------------
print("")
print("=== POPULATION BY UHI CLASS ===")
class_labels = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}
summary_rows = []
for c in [1, 2, 3, 4, 5]:
    class_mask = uhi_on_pop_grid == c
    pop_in_class = np.nansum(np.where(class_mask, population, 0))
    pct = (pop_in_class / total_population) * 100 if total_population > 0 else 0
    print("Class " + str(c) + " (" + class_labels[c] + "): population = " + str(int(pop_in_class)) + " (" + str(round(pct, 1)) + " percent)")
    summary_rows.append({
        "uhi_class": c,
        "uhi_label": class_labels[c],
        "population": int(pop_in_class),
        "percent_of_total": round(pct, 2)
    })

summary_df = pd.DataFrame(summary_rows)
csv_path = os.path.join(OUTPUT_DIR, "population_by_uhi_class.csv")
summary_df.to_csv(csv_path, index=False)
print("")
print("Saved: " + csv_path)

# --- Save rasters ---------------------------------------------------------------
print("")
print("Saving output rasters...")
out_profile = pop_profile.copy()
out_profile.update(dtype="float32", nodata=np.nan)

dist_path = os.path.join(OUTPUT_DIR, "distance_to_green_space_km.tif")
with rasterio.open(dist_path, "w", **out_profile) as dst:
    dst.write(distance_km.astype(np.float32), 1)
print("Saved: " + dist_path)

priority_path = os.path.join(OUTPUT_DIR, "priority_intervention_zones.tif")
with rasterio.open(priority_path, "w", **out_profile) as dst:
    dst.write(priority_zone_mask.astype(np.float32), 1)
print("Saved: " + priority_path)

# --- Map visualization --------------------------------------------------------
print("")
print("Generating equity map...")

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

im1 = axes[0].imshow(distance_km, cmap="YlGnBu_r", vmin=0, vmax=5)
axes[0].set_title("Distance to Nearest Green Space (km)", fontsize=13, fontweight="bold")
axes[0].axis("off")
plt.colorbar(im1, ax=axes[0], label="Distance (km)", fraction=0.04, pad=0.03)

priority_display = np.where(priority_zone_mask, 1, np.where(np.isnan(uhi_on_pop_grid), np.nan, 0))
cmap_priority = mcolors.ListedColormap(["#f0f0f0", "#b2182b"])
im2 = axes[1].imshow(priority_display, cmap=cmap_priority, vmin=0, vmax=1)
axes[1].set_title("Priority Intervention Zones\n(High Heat + Far from Green Space)", fontsize=13, fontweight="bold")
axes[1].axis("off")

plt.tight_layout()
map_path = os.path.join(OUTPUT_DIR, "green_space_equity_map.png")
plt.savefig(map_path, dpi=150, bbox_inches="tight")
print("Saved: " + map_path)
plt.close()

print("")
print("Step 4 complete.")
print("KEY FINDING: " + str(int(population_in_priority)) + " people (" + str(round(pct_population_at_risk, 1)) + " percent of study area population)")
print("live in zones with High/Very High heat AND no green space within 1km.")
print("This is your core equity finding for the project.")
