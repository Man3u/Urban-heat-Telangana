import streamlit as st
import pandas as pd
import json
import os
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Hyderabad Urban Heat & Green Space Equity",
    layout="wide",
    initial_sidebar_state="collapsed"
)

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

def load_json(name):
    with open(os.path.join(ASSETS_DIR, name)) as f:
        return json.load(f)

def asset_path(name):
    return os.path.join(ASSETS_DIR, name)

summary = load_json("summary_stats.json")
bounds = load_json("uhi_bounds.json")

# --- Header ------------------------------------------------------------------
st.title("Hyderabad Urban Heat Island & Green Space Equity")
st.markdown(
    "A GeoAI analysis of heat exposure and green space access across "
    "**" + summary["study_area"] + "**, built with Landsat, Sentinel-2, "
    "Google Earth Engine, and Python."
)

st.markdown("---")

# --- Key metrics ---------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Population (study area)", "{:,}".format(summary["total_population"]))
col2.metric("Population in High/Very High Heat", "{:,}".format(summary["high_heat_population"]), str(summary["pct_high_heat"]) + "% of total")
col3.metric("Population in Priority Zones", "{:,}".format(summary["population_in_priority_zones"]), str(summary["pct_priority_zones"]) + "% of total")
col4.metric("Avg. Distance to Green Space", str(summary["mean_distance_to_green_space_km"]) + " km")

st.markdown(
    "**Priority zones** are areas with High or Very High Urban Heat Island "
    "intensity that also sit more than 1km from any park, garden, or green space. "
    "These are the neighbourhoods most in need of climate adaptation investment."
)

st.markdown("---")

# --- Tabs ----------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Interactive Map",
    "Current Heat Map",
    "UHI Classification",
    "Green Space Equity",
    "12-Year Trend",
    "Population Breakdown"
])

with tab1:
    st.subheader("Interactive Urban Heat Island Map")
    st.markdown(
        "Zoom and pan across Hyderabad. Colors show UHI intensity class: "
        "blue (Very Low/Low) to red (Very High)."
    )

    center_lat = (bounds["north"] + bounds["south"]) / 2
    center_lon = (bounds["east"] + bounds["west"]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")

    folium.raster_layers.ImageOverlay(
        image=asset_path("uhi_overlay.png"),
        bounds=[[bounds["south"], bounds["west"]], [bounds["north"], bounds["east"]]],
        opacity=0.75,
        name="UHI Intensity"
    ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width=1200, height=600)

with tab2:
    st.subheader("Land Surface Temperature - Current (2022-2024 average)")
    st.image(asset_path("lst_current_map.png"), use_container_width=True)

with tab3:
    st.subheader("Urban Heat Island Intensity Classification")
    st.markdown("Classified using Natural Breaks (Jenks) on a combined LST/NDVI/NDBI index.")
    st.image(asset_path("uhi_index_map.png"), use_container_width=True)

with tab4:
    st.subheader("Green Space Equity Analysis")
    st.markdown(
        "Left: distance from every location to the nearest park or green space. "
        "Right: priority intervention zones where high heat overlaps with poor green space access."
    )
    st.image(asset_path("green_space_equity_map.png"), use_container_width=True)

with tab5:
    st.subheader("12-Year Land Surface Temperature Trend (2013-2024)")
    st.markdown(
        "Regionwide mean pre-monsoon LST computed directly from Earth Engine for each year. "
        "A sensitivity analysis excluding low-confidence years (fewer than 3 usable satellite "
        "scenes) is shown alongside the full dataset."
    )
    st.image(asset_path("lst_trend_chart.png"), use_container_width=True)

    trend_csv_path = asset_path("lst_trend_2013_2024.csv")
    if os.path.exists(trend_csv_path):
        trend_df = pd.read_csv(trend_csv_path)
        with st.expander("View raw yearly data"):
            st.dataframe(trend_df, use_container_width=True)

with tab6:
    st.subheader("Population by Heat Intensity Class")
    pop_csv_path = asset_path("population_by_uhi_class.csv")
    if os.path.exists(pop_csv_path):
        pop_df = pd.read_csv(pop_csv_path)
        color_map = {
            "Very Low": "#2166ac",
            "Low": "#67a9cf",
            "Moderate": "#fddbc7",
            "High": "#ef8a62",
            "Very High": "#b2182b"
        }
        fig = px.bar(
            pop_df,
            x="uhi_label",
            y="population",
            color="uhi_label",
            color_discrete_map=color_map,
            labels={"uhi_label": "UHI Intensity Class", "population": "Population"},
            title="Population Living in Each UHI Intensity Class"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pop_df, use_container_width=True)

st.markdown("---")

# --- Footer ----------------------------------------------------------------------
st.markdown(
    "**Methodology:** Landsat 8/9 (Land Surface Temperature), Sentinel-2 (NDVI/NDBI/NDWI), "
    "ESA WorldCover, WorldPop, and OpenStreetMap, processed via Google Earth Engine and Python "
    "(rasterio, geopandas, scipy, osmnx). Urban Heat Island classification uses Natural Breaks "
    "(Jenks). Green space equity uses Euclidean distance to nearest park, overlaid with "
    "gridded population data."
)

st.markdown(
    "Built by **Manu Chauhan Mudavath** - GIS & Remote Sensing Professional | "
    "[LinkedIn](https://linkedin.com/in/manu-chauhan-mudavath) | "
    "[GitHub](https://github.com/Man3u/Urban-heat-Telangana) | "
    "manuchauhanm76@gmail.com"
)
