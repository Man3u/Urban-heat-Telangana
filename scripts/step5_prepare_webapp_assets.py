import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds
import numpy as np
import pandas as pd
from PIL import Image
import json
import shutil
import os

BASE_DIR    = os.path.expanduser("~/Desktop/UrbanHeatHyderabad")
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
WEBAPP_DIR  = os.path.join(BASE_DIR, "web_app")
ASSETS_DIR  = os.path.join(WEBAPP_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

print("Preparing lightweight assets for the Streamlit web app...")
print("Assets folder: " + ASSETS_DIR)

# --- Copy PNG maps and CSVs (small files, safe to check into git) -----------
files_to_copy = [
    "lst_current_map.png",
    "uhi_index_map.png",
    "lst_trend_chart.png",
    "green_space_equity_map.png",
    "lst_trend_2013_2024.csv",
    "population_by_uhi_class.csv"
]

print("")
print("Copying PNGs and CSVs...")
for fname in files_to_copy:
    src_path = os.path.join(OUTPUT_DIR, fname)
    dst_path = os.path.join(ASSETS_DIR, fname)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print("Copied: " + fname)
    else:
        print("WARNING: not found, skipping: " + fname)

# --- Generate WGS84 reprojected UHI overlay for interactive map -------------
print("")
print("Generating WGS84 UHI overlay for interactive map...")

uhi_class_path = os.path.join(OUTPUT_DIR, "uhi_classified_current.tif")
dst_crs = "EPSG:4326"

with rasterio.open(uhi_class_path) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
    uhi_wgs84 = np.empty((height, width), dtype=np.float32)
    reproject(
        source=rasterio.band(src, 1),
        destination=uhi_wgs84,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest
    )

west, south, east, north = array_bounds(height, width, transform)
print("WGS84 bounds: west=" + str(round(west, 4)) + " south=" + str(round(south, 4)) + " east=" + str(round(east, 4)) + " north=" + str(round(north, 4)))

class_colors = {
    1: (33, 102, 172, 160),
    2: (103, 169, 207, 160),
    3: (253, 219, 199, 160),
    4: (239, 138, 98, 180),
    5: (178, 24, 43, 200),
}

rgba = np.zeros((height, width, 4), dtype=np.uint8)
for class_val, color in class_colors.items():
    mask = uhi_wgs84 == class_val
    rgba[mask] = color

nan_mask = np.isnan(uhi_wgs84)
rgba[nan_mask] = (0, 0, 0, 0)

overlay_img = Image.fromarray(rgba, mode="RGBA")
overlay_path = os.path.join(ASSETS_DIR, "uhi_overlay.png")
overlay_img.save(overlay_path)
print("Saved: " + overlay_path)

bounds_dict = {"west": west, "south": south, "east": east, "north": north}
bounds_path = os.path.join(ASSETS_DIR, "uhi_bounds.json")
with open(bounds_path, "w") as f:
    json.dump(bounds_dict, f, indent=2)
print("Saved: " + bounds_path)

# --- Compute summary statistics for the dashboard header --------------------
print("")
print("Computing summary statistics...")

pop_class_df = pd.read_csv(os.path.join(OUTPUT_DIR, "population_by_uhi_class.csv"))
total_population = int(pop_class_df["population"].sum())
high_heat_population = int(pop_class_df[pop_class_df["uhi_class"].isin([4, 5])]["population"].sum())
pct_high_heat = round((high_heat_population / total_population) * 100, 1)

with rasterio.open(os.path.join(OUTPUT_DIR, "priority_intervention_zones.tif")) as src:
    priority_mask = src.read(1)

population_path = os.path.join(DATA_DIR, "population_hyderabad_2020.tif")
with rasterio.open(population_path) as src:
    population = src.read(1).astype(np.float32)
    pop_nodata = src.nodata

if pop_nodata is not None:
    population = np.where(population == pop_nodata, 0, population)
population = np.where(population < 0, 0, population)

population_in_priority = float(np.nansum(np.where(priority_mask == 1, population, 0)))
pct_priority = round((population_in_priority / total_population) * 100, 1)

with rasterio.open(os.path.join(OUTPUT_DIR, "distance_to_green_space_km.tif")) as src:
    distance_km = src.read(1)
mean_distance_km = round(float(np.nanmean(distance_km)), 2)

trend_df = pd.read_csv(os.path.join(OUTPUT_DIR, "lst_trend_2013_2024.csv"))

summary = {
    "total_population": total_population,
    "high_heat_population": high_heat_population,
    "pct_high_heat": pct_high_heat,
    "population_in_priority_zones": int(population_in_priority),
    "pct_priority_zones": pct_priority,
    "mean_distance_to_green_space_km": mean_distance_km,
    "years_analysed": int(len(trend_df)),
    "study_area": "Hyderabad Metropolitan Region, Telangana, India"
}

summary_path = os.path.join(ASSETS_DIR, "summary_stats.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print("Saved: " + summary_path)
print("")
print("=== SUMMARY STATS ===")
for key, value in summary.items():
    print(str(key) + ": " + str(value))

print("")
print("Step 5 (asset prep) complete.")
print("Next: build web_app/app.py and run it locally with 'streamlit run app.py'")
