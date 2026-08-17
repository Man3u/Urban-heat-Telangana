import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR    = os.path.expanduser("~/Desktop/UrbanHeatHyderabad")
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LST_CURRENT_PATH = os.path.join(DATA_DIR, "lst_hyderabad_current_2022_2024.tif")

LST_MIN_VALID = 10.0
LST_MAX_VALID = 55.0

print("Loading LST Current 2022-2024 (Landsat 9, 3-year avg)...")
if not os.path.exists(LST_CURRENT_PATH):
    raise FileNotFoundError(
        "Could not find: " + LST_CURRENT_PATH + "\n" +
        "Make sure the file is downloaded and placed in the data folder."
    )

with rasterio.open(LST_CURRENT_PATH) as src:
    lst = src.read(1).astype(np.float32)
    profile = src.profile
    nodata = src.nodata

print("Raster shape: " + str(lst.shape))
print("CRS: " + str(profile['crs']))

if nodata is not None:
    lst = np.where(lst == nodata, np.nan, lst)

total_pixels = lst.size
invalid_mask = (lst < LST_MIN_VALID) | (lst > LST_MAX_VALID)
invalid_count = np.sum(invalid_mask & ~np.isnan(lst))
lst_clean = np.where(invalid_mask, np.nan, lst)

valid_pixels = np.sum(~np.isnan(lst_clean))
print("Total pixels: " + str(total_pixels))
print("Invalid or outlier pixels removed: " + str(invalid_count) + " (" + str(round(invalid_count/total_pixels*100, 2)) + " percent)")
print("Valid pixels remaining: " + str(valid_pixels))

print("")
print("=== Current Period (2022-2024) LST Statistics ===")
print("Mean: " + str(round(np.nanmean(lst_clean), 2)) + " C")
print("Median: " + str(round(np.nanmedian(lst_clean), 2)) + " C")
print("Min: " + str(round(np.nanmin(lst_clean), 2)) + " C")
print("Max: " + str(round(np.nanmax(lst_clean), 2)) + " C")
print("StdDev: " + str(round(np.nanstd(lst_clean), 2)) + " C")

p90 = np.nanpercentile(lst_clean, 90)
print("")
print("90th percentile threshold: " + str(round(p90, 2)) + " C")
print("Pixels at or above this threshold represent the hottest zones -")
print("these will form the core of the Urban Heat Island index in Step 3.")

profile.update(dtype="float32", nodata=np.nan)
clean_path = os.path.join(OUTPUT_DIR, "lst_current_clean.tif")
with rasterio.open(clean_path, "w", **profile) as dst:
    dst.write(lst_clean.astype(np.float32), 1)
print("")
print("Saved cleaned raster: " + clean_path)

print("Generating spatial heat map...")

vmin = np.nanpercentile(lst_clean, 2)
vmax = np.nanpercentile(lst_clean, 98)

fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(lst_clean, cmap="inferno", vmin=vmin, vmax=vmax)
ax.set_title("Hyderabad Land Surface Temperature - Current (2022-2024 average)", fontsize=13, fontweight="bold")
ax.axis("off")
cbar = plt.colorbar(im, ax=ax, label="Temperature (C)", fraction=0.04, pad=0.03)

plt.tight_layout()
map_path = os.path.join(OUTPUT_DIR, "lst_current_map.png")
plt.savefig(map_path, dpi=150, bbox_inches="tight")
print("Saved: " + map_path)
plt.close()

print("")
print("Step 2 complete.")
print("Next: run step3_uhi_index.py to build the combined Urban Heat Island index.")
