import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import jenkspy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

BASE_DIR   = os.path.expanduser("~/Desktop/UrbanHeatHyderabad")
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LST_PATH = os.path.join(OUTPUT_DIR, "lst_current_clean.tif")
S2_PATH  = os.path.join(DATA_DIR, "s2_indices_hyderabad_current_2022_2024.tif")

NDWI_WATER_THRESHOLD = 0.1
JENKS_SAMPLE_SIZE = 20000

print("Loading cleaned LST raster (reference grid)...")
with rasterio.open(LST_PATH) as src:
    lst = src.read(1).astype(np.float32)
    lst_profile = src.profile
    lst_transform = src.transform
    lst_crs = src.crs
    lst_shape = (src.height, src.width)

print("LST shape: " + str(lst_shape))
print("Pixel size: " + str(round(lst_transform.a, 1)) + " m")

print("")
print("Loading Sentinel-2 indices raster (NDVI, NDBI, NDWI)...")
if not os.path.exists(S2_PATH):
    raise FileNotFoundError(
        "Could not find: " + S2_PATH + "\n" +
        "Make sure s2_indices_hyderabad_current_2022_2024.tif is in the data folder."
    )

with rasterio.open(S2_PATH) as src:
    s2_bands = src.read()
    s2_transform = src.transform
    s2_crs = src.crs
    band_count = src.count

print("Sentinel-2 raster shape: " + str(s2_bands.shape))
print("Band count: " + str(band_count) + " (expected order: NDVI, NDBI, NDWI)")

print("")
print("Resampling Sentinel-2 bands onto LST grid...")

def resample_to_lst_grid(src_band, src_transform, src_crs):
    dst = np.empty(lst_shape, dtype=np.float32)
    reproject(
        source=src_band,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=lst_transform,
        dst_crs=lst_crs,
        resampling=Resampling.average
    )
    return dst

ndvi = resample_to_lst_grid(s2_bands[0], s2_transform, s2_crs)
ndbi = resample_to_lst_grid(s2_bands[1], s2_transform, s2_crs)
ndwi = resample_to_lst_grid(s2_bands[2], s2_transform, s2_crs)

print("Resampling complete.")
print("NDVI range: " + str(round(np.nanmin(ndvi), 3)) + " to " + str(round(np.nanmax(ndvi), 3)))
print("NDBI range: " + str(round(np.nanmin(ndbi), 3)) + " to " + str(round(np.nanmax(ndbi), 3)))
print("NDWI range: " + str(round(np.nanmin(ndwi), 3)) + " to " + str(round(np.nanmax(ndwi), 3)))

print("")
print("Masking water bodies (NDWI > " + str(NDWI_WATER_THRESHOLD) + ")...")
water_mask = ndwi > NDWI_WATER_THRESHOLD
water_pixel_count = np.sum(water_mask)
print("Water pixels masked: " + str(water_pixel_count))

lst_masked  = np.where(water_mask, np.nan, lst)
ndvi_masked = np.where(water_mask, np.nan, ndvi)
ndbi_masked = np.where(water_mask, np.nan, ndbi)

def normalize(array):
    a_min = np.nanmin(array)
    a_max = np.nanmax(array)
    return (array - a_min) / (a_max - a_min)

print("")
print("Normalizing LST, NDVI, NDBI to 0-1 range...")
lst_norm  = normalize(lst_masked)
ndvi_norm = normalize(ndvi_masked)
ndbi_norm = normalize(ndbi_masked)

print("Computing UHI score = norm(LST) - norm(NDVI) + norm(NDBI)...")
uhi_score = lst_norm - ndvi_norm + ndbi_norm
uhi_score_norm = normalize(uhi_score)

# --- Classify using Natural Breaks (Jenks) ----------------------------------
# Same method ArcGIS Pro uses by default for choropleth symbology.
# Unlike quantile classification (which always forces 5 equal-sized groups
# no matter what the data looks like), Jenks finds actual clusters in the
# data - so it reveals genuine hotspot concentration instead of an
# artificially even split.
print("")
print("Classifying into 5 UHI intensity classes (Natural Breaks / Jenks)...")

valid_mask = ~np.isnan(uhi_score_norm)
valid_scores = uhi_score_norm[valid_mask]

# Jenks is computed on a random sample for speed (standard practice -
# ArcGIS itself samples large rasters for classification), then the
# resulting breakpoints are applied to the full dataset.
rng = np.random.default_rng(42)
sample_size = min(JENKS_SAMPLE_SIZE, valid_scores.size)
sample_indices = rng.choice(valid_scores.size, size=sample_size, replace=False)
sample = valid_scores[sample_indices]

breaks = jenkspy.jenks_breaks(sample.tolist(), n_classes=5)
print("Jenks natural breaks: " + str([round(b, 3) for b in breaks]))

uhi_class = np.full(uhi_score_norm.shape, np.nan)
class_vals = np.digitize(valid_scores, breaks[1:-1]) + 1
uhi_class[valid_mask] = class_vals

class_labels = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}

pixel_area_km2 = (abs(lst_transform.a) * abs(lst_transform.e)) / 1_000_000
print("")
print("Pixel area: " + str(round(pixel_area_km2, 5)) + " sq km")
print("")
print("=== UHI CLASS AREA SUMMARY (Natural Breaks) ===")
total_valid = np.sum(~np.isnan(uhi_class))
for c in [1, 2, 3, 4, 5]:
    count = np.sum(uhi_class == c)
    area_km2 = count * pixel_area_km2
    pct = (count / total_valid) * 100 if total_valid > 0 else 0
    print("Class " + str(c) + " (" + class_labels[c] + "): " + str(count) + " pixels, " + str(round(area_km2, 2)) + " sq km, " + str(round(pct, 1)) + " percent")

print("")
print("Saving UHI rasters...")
out_profile = lst_profile.copy()
out_profile.update(dtype="float32", nodata=np.nan)

uhi_score_path = os.path.join(OUTPUT_DIR, "uhi_score_current.tif")
with rasterio.open(uhi_score_path, "w", **out_profile) as dst:
    dst.write(uhi_score_norm.astype(np.float32), 1)
print("Saved: " + uhi_score_path)

uhi_class_path = os.path.join(OUTPUT_DIR, "uhi_classified_current.tif")
with rasterio.open(uhi_class_path, "w", **out_profile) as dst:
    dst.write(uhi_class.astype(np.float32), 1)
print("Saved: " + uhi_class_path)

print("")
print("Generating UHI classification map...")

class_colors = ["#2166ac", "#67a9cf", "#fddbc7", "#ef8a62", "#b2182b"]
cmap = mcolors.ListedColormap(class_colors)
bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(uhi_class, cmap=cmap, norm=norm)
ax.set_title("Hyderabad Urban Heat Island Intensity - Current (2022-2024)", fontsize=13, fontweight="bold")
ax.axis("off")

cbar = plt.colorbar(im, ax=ax, ticks=[1, 2, 3, 4, 5], fraction=0.04, pad=0.03)
cbar.ax.set_yticklabels(["Very Low", "Low", "Moderate", "High", "Very High"])
cbar.set_label("UHI Intensity Class (Natural Breaks)")

plt.tight_layout()
map_path = os.path.join(OUTPUT_DIR, "uhi_index_map.png")
plt.savefig(map_path, dpi=150, bbox_inches="tight")
print("Saved: " + map_path)
plt.close()

print("")
print("Step 3 complete.")
print("Next: run step4_landcover_classification.py to build the AI land cover model,")
print("then step5_equity_analysis.py to overlay population and green space access.")
