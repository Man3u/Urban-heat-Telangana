import ee
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

print("Initialising Google Earth Engine...")
ee.Initialize(project='urban-heat-hyderabad')
print("  GEE initialised.")

BASE_DIR   = os.path.expanduser("~/Desktop/UrbanHeatHyderabad")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

hyderabad = ee.Geometry.Rectangle([78.20, 17.15, 78.75, 17.65])
YEARS = list(range(2013, 2025))

# Relaxed cloud threshold and widened pre-monsoon window to get more
# scenes per year - a 2-3 scene composite is too unstable to trust,
# especially at the endpoints of a regression where it has high leverage.
CLOUD_COVER_MAX = 50
SEASON_START_MD = "-02-15"
SEASON_END_MD   = "-05-31"
MIN_SCENES_FOR_CONFIDENCE = 3

def mask_landsat_clouds(image):
    qa = image.select('QA_PIXEL')
    cloud_bit  = 1 << 3
    shadow_bit = 1 << 4
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(shadow_bit).eq(0))
    return image.updateMask(mask)

def compute_lst(image):
    thermal = image.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST_C')
    return image.addBands(thermal)

print("Computing regionwide mean LST for 2013-2024...")
print("(Cloud filter relaxed to " + str(CLOUD_COVER_MAX) + "%, season widened to Feb15-May31 for more scenes per year)")
print("")

results = []
for year in YEARS:
    start = str(year) + SEASON_START_MD
    end   = str(year) + SEASON_END_MD
    if year >= 2022:
        collection_id = "LANDSAT/LC09/C02/T1_L2"
    else:
        collection_id = "LANDSAT/LC08/C02/T1_L2"

    coll = (ee.ImageCollection(collection_id)
            .filterBounds(hyderabad)
            .filterDate(start, end)
            .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER_MAX))
            .map(mask_landsat_clouds)
            .map(compute_lst))

    scene_count = coll.size().getInfo()
    if scene_count == 0:
        print(str(year) + ": no usable scenes found - skipping")
        continue

    composite = coll.select('LST_C').median().clip(hyderabad)

    stats_dict = composite.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=hyderabad,
        scale=200,
        maxPixels=1e9,
        bestEffort=True
    ).getInfo()

    mean_lst = stats_dict.get('LST_C')
    if mean_lst is None:
        print(str(year) + ": could not compute mean - skipping")
        continue

    confidence_flag = "OK" if scene_count >= MIN_SCENES_FOR_CONFIDENCE else "LOW CONFIDENCE"
    print(str(year) + ": mean LST = " + str(round(mean_lst, 2)) + " C (from " + str(scene_count) + " scenes) [" + confidence_flag + "]")
    results.append({
        "year": year,
        "mean_lst_c": mean_lst,
        "scene_count": scene_count,
        "low_confidence": scene_count < MIN_SCENES_FOR_CONFIDENCE
    })

df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "lst_trend_2013_2024.csv")
df.to_csv(csv_path, index=False)
print("")
print("Saved raw yearly data: " + csv_path)

def fit_and_report(data, label):
    x = data["year"].values
    y = data["mean_lst_c"].values
    if len(x) < 3:
        print(label + ": not enough years to fit a trend")
        return None
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    warming_per_decade = slope * 10

    print("")
    print("=== " + label + " ===")
    print("Years analysed: " + str(len(data)) + " (" + str(x.min()) + "-" + str(x.max()) + ")")
    print("Slope: " + str(round(slope, 4)) + " C/year")
    print("Warming rate: " + str(round(warming_per_decade, 2)) + " C per decade")
    print("R-squared: " + str(round(r_squared, 3)))
    print("P-value: " + str(round(p_value, 4)))

    if p_value < 0.05:
        print("Result: STATISTICALLY SIGNIFICANT (p < 0.05)")
        if slope > 0:
            print("Direction: WARMING")
        else:
            print("Direction: COOLING")
    else:
        print("Result: NOT statistically significant (p >= 0.05)")
        print("Trend direction present but not strong enough to be confident it")
        print("isn't due to year-to-year weather variability alone.")

    return {"slope": slope, "intercept": intercept, "r_squared": r_squared,
            "p_value": p_value, "x": x, "y": y, "label": label}

print("\nFitting trend - ALL YEARS (including low-confidence)...")
result_all = fit_and_report(df, "ALL YEARS (n=" + str(len(df)) + ")")

df_confident = df[df["low_confidence"] == False].copy()
print("\nFitting trend - HIGH CONFIDENCE YEARS ONLY (excluding years with <" + str(MIN_SCENES_FOR_CONFIDENCE) + " scenes)...")
result_confident = fit_and_report(df_confident, "HIGH CONFIDENCE (n=" + str(len(df_confident)) + ")")

low_conf_years = df[df["low_confidence"] == True]["year"].tolist()
if low_conf_years:
    print("\nExcluded low-confidence years: " + str(low_conf_years))

print("\nGenerating trend chart...")
fig, ax = plt.subplots(figsize=(11, 6.5))

colors = ["#999999" if lc else "#d55e00" for lc in df["low_confidence"]]
ax.scatter(df["year"], df["mean_lst_c"], color=colors, s=90, zorder=3,
           label="Annual mean LST (grey = low confidence, <3 scenes)")

if result_all is not None:
    trend_line_all = result_all["slope"] * result_all["x"] + result_all["intercept"]
    label_all = "All years: " + str(round(result_all["slope"] * 10, 2)) + " C/decade (R2=" + str(round(result_all["r_squared"], 2)) + ", p=" + str(round(result_all["p_value"], 3)) + ")"
    ax.plot(result_all["x"], trend_line_all, color="#0072b2", linewidth=2, zorder=2, label=label_all)

if result_confident is not None:
    trend_line_conf = result_confident["slope"] * result_confident["x"] + result_confident["intercept"]
    label_conf = "High-confidence only: " + str(round(result_confident["slope"] * 10, 2)) + " C/decade (R2=" + str(round(result_confident["r_squared"], 2)) + ", p=" + str(round(result_confident["p_value"], 3)) + ")"
    ax.plot(result_confident["x"], trend_line_conf, color="#009e73", linewidth=2, linestyle="--", zorder=2, label=label_conf)

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Mean Land Surface Temperature (C)", fontsize=12)
ax.set_title("Hyderabad Pre-Monsoon Land Surface Temperature Trend (2013-2024)", fontsize=13, fontweight="bold")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)
ax.set_xticks(df["year"])
ax.set_xticklabels(df["year"], rotation=45)

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "lst_trend_chart.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
print("Saved: " + chart_path)
plt.close()

print("\nStep 1b complete.")
print("Report BOTH trend lines in your write-up for transparency - it shows")
print("methodological rigor (sensitivity analysis) rather than picking whichever")
print("number looks best.")
print("Next: run step2_lst_processing.py for the spatial heat map.")
