# Urban Heat Island & Green Space Equity — Hyderabad

**Live dashboard:** https://urban-heat-telangana-jxv4ijcstuemzkzrz2k3gc.streamlit.app
**Repository:** https://github.com/Man3u/Urban-heat-Telangana

A GeoAI analysis of heat exposure and green space accessibility across the Hyderabad Metropolitan Region, Telangana, India. The project combines Landsat thermal imagery, Sentinel-2 spectral indices, population data, and OpenStreetMap green space locations to identify which communities face the highest combined burden of urban heat and poor green space access — and delivers the findings through a public, interactive web dashboard.

## Key Findings

Across a study population of 8,376,303 people, 63.8% (5,340,189 people) live in areas classified as High or Very High Urban Heat Island intensity. Of these, 2,271,067 people (27.1% of the total population) live in **priority intervention zones** — areas that are both high-heat and more than 1km from the nearest park or green space. The average distance to green space across the study area is 3.02 km.

A 12-year land surface temperature time series (2013–2024) was analysed for a long-term trend. Using all 12 years, the trend was statistically significant but showed cooling (-3.68°C/decade, p=0.043); however, a sensitivity analysis excluding 2013 — which had only 2 usable cloud-free satellite scenes and is not a reliable annual estimate — found no statistically significant trend in either direction (-2.0°C/decade, p=0.219, n=11). This is reported transparently rather than presenting the more dramatic but less reliable result: it reflects genuine year-to-year weather variability rather than a confirmed long-term warming or cooling signal, and demonstrates why single- or few-scene annual composites should not be over-interpreted.

## Methodology

**Land Surface Temperature.** Landsat 8 (2013–2021) and Landsat 9 (2022–2024) thermal band data was pulled from Google Earth Engine, cloud-masked using the QA_PIXEL band, and converted to Celsius using the USGS Collection 2 Level 2 scale factors. A pre-monsoon (Feb 15–May 31) seasonal window was used for both the current-period spatial analysis and the annual trend series, since this is the hottest and most comparable period year to year in Telangana.

**Urban Heat Island Index.** NDVI, NDBI, and NDWI were computed from Sentinel-2 surface reflectance imagery and resampled onto the LST grid. Water bodies (NDWI > 0.1) were masked out before analysis. LST, NDVI, and NDBI were each normalised to a 0–1 range and combined into a single UHI score: `normalize(LST) − normalize(NDVI) + normalize(NDBI)`. The result was classified into 5 intensity classes using Natural Breaks (Jenks) — the same classification method used by default in ArcGIS Pro — rather than quantile classification, since quantiles force an artificially even split regardless of the underlying data distribution.

**Trend Analysis.** Rather than comparing two arbitrary time periods (which proved fragile — an initial 2015-vs-2024 comparison was skewed by the well-documented 2015 India heatwave), a 12-year annual time series of regionwide mean LST was built directly via Earth Engine's `reduceRegion`, and a linear regression was fit using scipy. Results are reported both including and excluding low-confidence years (fewer than 3 usable satellite scenes), following standard practice for climate trend sensitivity analysis.

**Green Space Equity.** Parks, gardens, recreation grounds, and forested areas were queried live from OpenStreetMap via `osmnx` (1,860 features found, 1,670 valid polygons). A Euclidean distance transform was computed from every location in the city to the nearest green space. This was combined with the UHI classification and WorldPop 2020 gridded population data to identify and quantify population living in priority intervention zones.

**Web Dashboard.** A Streamlit application was built to make the findings publicly accessible without requiring GIS software. It includes an interactive Folium map of the UHI classification, static maps for each analysis stage, and interactive charts of the population and trend data, all backed by lightweight pre-processed assets (PNGs, CSVs, JSON) rather than raw rasters, keeping the deployed app fast and dependency-light.

## Data Sources

| Data | Source | Resolution | Purpose |
|---|---|---|---|
| Landsat 8/9 (Collection 2 Level 2) | USGS via Google Earth Engine | 30m | Land Surface Temperature |
| Sentinel-2 SR Harmonized | Copernicus via Google Earth Engine | 10m | NDVI, NDBI, NDWI |
| ESA WorldCover 2021 | Google Earth Engine | 10m | Land cover context |
| WorldPop 2020 | Google Earth Engine | 100m | Population density |
| OpenStreetMap | via osmnx | vector | Parks and green space |

## Repository Structure

```
scripts/
  step1_gee_data_collection.py       Earth Engine export: LST, Sentinel-2, WorldCover, WorldPop
  step1b_lst_trend_analysis.py       12-year LST trend with sensitivity analysis
  step2_lst_processing.py            Current-period LST cleaning and spatial mapping
  step3_uhi_index.py                 UHI index (LST/NDVI/NDBI) with Jenks classification
  step4_green_space_equity.py        OSM green space distance + population equity analysis
  step5_prepare_webapp_assets.py     Prepares lightweight assets for the Streamlit app
web_app/
  app.py                             Streamlit dashboard
  requirements.txt
  assets/                            PNGs, CSVs, and JSON summaries used by the app
```

Raw satellite data and full-resolution output rasters are excluded from version control (see `.gitignore`) due to file size — the pipeline is fully reproducible by running the scripts in order against a fresh Earth Engine export.

## Tech Stack

Python (rasterio, geopandas, numpy, pandas, scipy, osmnx, jenkspy), Google Earth Engine, Streamlit, Folium, Plotly, Git/GitHub.

## Author

**Manu Chauhan Mudavath**
MSc Computer Science — University of Waikato, New Zealand
MSc Information Systems — University of West London, UK
BTech — MGIT Hyderabad, India

[LinkedIn](https://linkedin.com/in/manu-chauhan-mudavath) · manuchauhanm76@gmail.com
