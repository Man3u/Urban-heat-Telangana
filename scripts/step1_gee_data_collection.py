import ee

print("Initialising Google Earth Engine...")
ee.Initialize(project='urban-heat-hyderabad')
print("  GEE initialised.")

hyderabad = ee.Geometry.Rectangle([78.20, 17.15, 78.75, 17.65])

DRIVE_FOLDER = "UrbanHeat_Hyderabad"
CRS          = "EPSG:32644"

BASELINE_YEARS = [2014, 2015, 2016]
CURRENT_YEARS  = [2022, 2023, 2024]
SEASON_START_MD = "-03-01"
SEASON_END_MD   = "-05-31"

def mask_landsat_clouds(image):
    qa = image.select('QA_PIXEL')
    cloud_bit  = 1 << 3
    shadow_bit = 1 << 4
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(shadow_bit).eq(0))
    return image.updateMask(mask)

def compute_lst(image):
    thermal = image.select('ST_B10') \
        .multiply(0.00341802).add(149.0) \
        .subtract(273.15) \
        .rename('LST_C')
    return image.addBands(thermal)

def get_multiyear_lst(collection_id, years, region):
    yearly_images = []
    for year in years:
        start = f"{year}{SEASON_START_MD}"
        end   = f"{year}{SEASON_END_MD}"
        coll = (ee.ImageCollection(collection_id)
                .filterBounds(region)
                .filterDate(start, end)
                .filter(ee.Filter.lt('CLOUD_COVER', 30))
                .map(mask_landsat_clouds)
                .map(compute_lst))
        yearly_median = coll.select('LST_C').median()
        yearly_images.append(yearly_median)
    collection_of_years = ee.ImageCollection.fromImages(yearly_images)
    return collection_of_years.mean().clip(region)

print(f"\nBuilding LST baseline composite ({BASELINE_YEARS}, Landsat 8)...")
lst_baseline = get_multiyear_lst("LANDSAT/LC08/C02/T1_L2", BASELINE_YEARS, hyderabad)

print(f"Building LST current composite ({CURRENT_YEARS}, Landsat 9)...")
lst_current = get_multiyear_lst("LANDSAT/LC09/C02/T1_L2", CURRENT_YEARS, hyderabad)

for label, image in [("baseline_2014_2016", lst_baseline), ("current_2022_2024", lst_current)]:
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"LST_Hyderabad_{label}",
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"lst_hyderabad_{label}",
        region=hyderabad,
        scale=30,
        crs=CRS,
        maxPixels=1e13,
        fileFormat="GeoTIFF"
    )
    task.start()
    print(f"  LST {label} export started. Task ID: {task.id}")

S2_BASELINE_YEARS = [2015, 2016]
S2_CURRENT_YEARS  = [2022, 2023, 2024]

def add_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    return image.addBands([ndvi, ndbi, ndwi])

def get_multiyear_sentinel(years, region):
    yearly_images = []
    for year in years:
        start = f"{year}{SEASON_START_MD}"
        end   = f"{year}{SEASON_END_MD}"
        coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(region)
                .filterDate(start, end)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))
        if coll.size().getInfo() == 0:
            print(f"    Warning: no Sentinel-2 scenes found for {year}, skipping.")
            continue
        yearly_median = add_indices(coll.median())
        yearly_images.append(yearly_median.select(['NDVI', 'NDBI', 'NDWI']))
    collection_of_years = ee.ImageCollection.fromImages(yearly_images)
    return collection_of_years.mean().clip(region)

print(f"\nBuilding Sentinel-2 indices baseline composite ({S2_BASELINE_YEARS})...")
s2_baseline = get_multiyear_sentinel(S2_BASELINE_YEARS, hyderabad)

print(f"Building Sentinel-2 indices current composite ({S2_CURRENT_YEARS})...")
s2_current = get_multiyear_sentinel(S2_CURRENT_YEARS, hyderabad)

for label, image in [("baseline_2015_2016", s2_baseline), ("current_2022_2024", s2_current)]:
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"Sentinel2_Indices_Hyderabad_{label}",
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"s2_indices_hyderabad_{label}",
        region=hyderabad,
        scale=10,
        crs=CRS,
        maxPixels=1e13,
        fileFormat="GeoTIFF"
    )
    task.start()
    print(f"  Sentinel-2 indices {label} export started. Task ID: {task.id}")

print("\nExporting Sentinel-2 raw bands (2024) for U-Net training...")
s2_2024_raw = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterBounds(hyderabad)
               .filterDate("2024-03-01", "2024-05-31")
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
               .median()
               .clip(hyderabad))

task_bands = ee.batch.Export.image.toDrive(
    image=s2_2024_raw.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']),
    description="Sentinel2_RawBands_Hyderabad_2024",
    folder=DRIVE_FOLDER,
    fileNamePrefix="s2_rawbands_hyderabad_2024",
    region=hyderabad,
    scale=10,
    crs=CRS,
    maxPixels=1e13,
    fileFormat="GeoTIFF"
)
task_bands.start()
print(f"  Raw bands export started. Task ID: {task_bands.id}")

print("\nExporting ESA WorldCover 2021 (training labels)...")
worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().clip(hyderabad)

task_lulc = ee.batch.Export.image.toDrive(
    image=worldcover,
    description="WorldCover_Hyderabad_2021",
    folder=DRIVE_FOLDER,
    fileNamePrefix="worldcover_hyderabad_2021",
    region=hyderabad,
    scale=10,
    crs=CRS,
    maxPixels=1e13,
    fileFormat="GeoTIFF"
)
task_lulc.start()
print(f"  WorldCover export started. Task ID: {task_lulc.id}")

print("\nExporting WorldPop 2020 population density...")
worldpop = (ee.ImageCollection("WorldPop/GP/100m/pop")
            .filter(ee.Filter.eq("year", 2020))
            .filter(ee.Filter.eq("country", "IND"))
            .first()
            .clip(hyderabad))

task_pop = ee.batch.Export.image.toDrive(
    image=worldpop,
    description="Population_Hyderabad_2020",
    folder=DRIVE_FOLDER,
    fileNamePrefix="population_hyderabad_2020",
    region=hyderabad,
    scale=100,
    crs=CRS,
    maxPixels=1e13,
    fileFormat="GeoTIFF"
)
task_pop.start()
print(f"  Population export started. Task ID: {task_pop.id}")

print("\n=== ALL EXPORT TASKS SUBMITTED ===")
print("Monitor progress at: https://code.earthengine.google.com/tasks")
print(f"Files will appear in Google Drive > {DRIVE_FOLDER} folder\n")
print("Next step: download these to your Mac (data/ folder), then run")
print("the updated step2_lst_processing.py")
