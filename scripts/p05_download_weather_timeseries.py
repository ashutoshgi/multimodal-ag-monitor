
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Extract daily weather/environment time series by cleaned zone polygons.

Datasets:
- GRIDMET
- CHIRPS Daily
- ERA5-Land Daily Aggregated

Outputs:
- one CSV per zone
- one combined CSV for all zones

Key improvement in this version:
- uses buffered geometry for coarse weather datasets so small farm polygons
  still return valid values

"""

import os
import json
from datetime import datetime

import ee
import geopandas as gpd
import pandas as pd
import numpy as np

from eo_config import OUTPUT_DIR, FIELD_NAME, EE_PROJECT, cleaned_shapefile_path, START_YEAR, END_YEAR, WEATHER_START_YEAR, WEATHER_END_YEAR, GRIDMET_BUFFER_M, CHIRPS_BUFFER_M, ERA5_BUFFER_M, LANDSAT_START_YEAR, LANDSAT_END_YEAR, LANDSAT_DOWNLOAD_IMAGES

# =====================================================
# USER SETTINGS
# =====================================================
#Hare Krishna
#Hare Krishna
#Hare Krishna

BASE_DIR = OUTPUT_DIR
CLEANED_SHP = cleaned_shapefile_path()
OUTPUT_DIR = os.path.join(BASE_DIR, "Weather_TimeSeries")
PER_ZONE_DIR = os.path.join(OUTPUT_DIR, "per_zone_csv")


# OUTPUT_DIR = os.path.join(BASE_DIR, "Weather_TimeSeries")
# PER_ZONE_DIR = os.path.join(OUTPUT_DIR, "per_zone_csv")

# OUTPUT_DIR = os.path.join(BASE_DIR, "Weather_TimeSeries")
# PER_ZONE_DIR = os.path.join(OUTPUT_DIR, "per_zone_csv")

# EE_PROJECT loaded from config

START_DATE = f"{WEATHER_START_YEAR}-01-01"
if WEATHER_END_YEAR >= datetime.now().year:
    END_DATE = datetime.now().strftime("%Y-%m-%d")
else:
    END_DATE = f"{WEATHER_END_YEAR}-12-31"

CHUNK_YEARS = 1

GRIDMET_SCALE = 4638
CHIRPS_SCALE = 5566
ERA5_SCALE = 11132

# Buffer distances for small fields / coarse grids
# Buffer distances are loaded from config/config.yaml

# Optional fallback: if a dataset still returns mostly nulls, try a larger buffer
USE_FALLBACK_BUFFER = True
GRIDMET_BUFFER_M_FALLBACK = 8000
CHIRPS_BUFFER_M_FALLBACK = 9000
ERA5_BUFFER_M_FALLBACK = 16000

# =====================================================
# HELPERS
# =====================================================
def make_safe_name(text):
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(text))


def detect_source_name_column(gdf):
    for col in ["source_name", "source_nam", "source", "name", "Name", "NAME"]:
        if col in gdf.columns:
            return col
    return None


def shapely_to_ee_geometry(geom):
    geojson_geom = json.loads(
        gpd.GeoSeries([geom], crs="EPSG:4326").to_json()
    )["features"][0]["geometry"]
    return ee.Geometry(geojson_geom)


def init_ee():
    try:
        ee.Initialize(project=EE_PROJECT) if EE_PROJECT else ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT) if EE_PROJECT else ee.Initialize()


def build_zone_gdf(shp_path):
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Cleaned shapefile not found:\n{shp_path}")

    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("Cleaned shapefile is empty.")
    if gdf.crs is None:
        raise ValueError("Cleaned shapefile has no CRS.")

    gdf = gdf.to_crs(epsg=4326)

    required = ["zone_id", "part_id", "geometry"]
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns in cleaned shapefile: {missing}")

    source_col = detect_source_name_column(gdf)
    if source_col is None:
        gdf["source_label"] = gdf.apply(
            lambda r: f"zone_{int(r['zone_id']):02d}_part_{int(r['part_id'])}",
            axis=1
        )
    else:
        gdf["source_label"] = gdf[source_col].astype(str)

    return gdf


def buffer_geometry_meters(geom, meters):
    """
    Buffer polygon outward in meters using EPSG:3857, then return to EPSG:4326.
    """
    if geom is None or geom.is_empty:
        return geom
    if meters is None or meters <= 0:
        return geom

    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs="EPSG:4326").to_crs(epsg=3857)
    buffered = gdf.geometry.iloc[0].buffer(meters)
    return gpd.GeoDataFrame({"id": [1]}, geometry=[buffered], crs="EPSG:3857").to_crs(epsg=4326).geometry.iloc[0]


def count_non_null_values(df, ignore_cols=None):
    if df is None or df.empty:
        return 0
    if ignore_cols is None:
        ignore_cols = ["date", "dataset", "zone_id", "part_id", "source_name"]

    cols = [c for c in df.columns if c not in ignore_cols]
    total = 0
    for c in cols:
        total += int(pd.to_numeric(df[c], errors="coerce").notna().sum())
    return total


# =====================================================
# DATASET PREP
# =====================================================
def prep_gridmet(img):
    out = ee.Image.cat([
        img.select("pr").rename("gridmet_pr_mm"),
        img.select("tmmn").subtract(273.15).rename("gridmet_tmmn_C"),
        img.select("tmmx").subtract(273.15).rename("gridmet_tmmx_C"),
        img.select("sph").rename("gridmet_sph"),
        img.select("srad").rename("gridmet_srad_Wm2"),
        img.select("vs").rename("gridmet_wind_ms"),
        img.select("vpd").rename("gridmet_vpd_kPa"),
        img.select("eto").rename("gridmet_eto_mm")
    ])
    return out.copyProperties(img, ["system:time_start"])


def prep_chirps(img):
    out = img.select("precipitation").rename("chirps_precip_mm")
    return out.copyProperties(img, ["system:time_start"])


def prep_era5(img):
    out = ee.Image.cat([
        img.select("temperature_2m").subtract(273.15).rename("era5l_t2m_C"),
        img.select("dewpoint_temperature_2m").subtract(273.15).rename("era5l_d2m_C"),
        img.select("surface_net_solar_radiation_sum").rename("era5l_net_solar_Jm2"),
        img.select("total_precipitation_sum").multiply(1000.0).rename("era5l_total_precip_mm"),
        img.select("u_component_of_wind_10m").rename("era5l_u10_ms"),
        img.select("v_component_of_wind_10m").rename("era5l_v10_ms"),
        img.select("volumetric_soil_water_layer_1").rename("era5l_vswc_l1_m3m3"),
        img.select("volumetric_soil_water_layer_2").rename("era5l_vswc_l2_m3m3"),
        img.select("soil_temperature_level_1").subtract(273.15).rename("era5l_stl1_C"),
        img.select("soil_temperature_level_2").subtract(273.15).rename("era5l_stl2_C")
    ])
    return out.copyProperties(img, ["system:time_start"])


# =====================================================
# FEATURE EXTRACTION
# =====================================================
def imagecollection_to_zone_timeseries(ic, ee_geom, scale, dataset_name,
                                       start_date, end_date, chunk_years=1):
    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year

    all_rows = []

    for y in range(start_year, end_year + 1, chunk_years):
        chunk_start = f"{y:04d}-01-01"
        chunk_end_year = min(y + chunk_years - 1, end_year)

        if chunk_end_year == end_year:
            chunk_end = end_date
        else:
            chunk_end = f"{chunk_end_year + 1:04d}-01-01"

        ic_chunk = ic.filterDate(chunk_start, chunk_end)

        def per_image(img):
            date = img.date().format("YYYY-MM-dd")
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee_geom,
                scale=scale,
                maxPixels=1e13
            )
            return ee.Feature(None, stats.set("date", date).set("dataset", dataset_name))

        fc = ee.FeatureCollection(ic_chunk.map(per_image))
        info = fc.getInfo()

        rows = [feat["properties"] for feat in info["features"]]
        if rows:
            all_rows.extend(rows)

        print(f"    {dataset_name}: pulled {len(rows)} rows for {chunk_start} to {chunk_end}")

    df = pd.DataFrame(all_rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

    return df


def merge_on_date(dfs):
    valid = [d for d in dfs if d is not None and not d.empty]
    if not valid:
        return pd.DataFrame()

    merged = valid[0].copy()
    for d in valid[1:]:
        cols_to_use = [c for c in d.columns if c not in ["dataset"]]
        d2 = d[cols_to_use].copy()
        merged = merged.merge(d2, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def try_dataset_with_buffers(ic, geom, scale, dataset_name, start_date, end_date,
                             primary_buffer_m, fallback_buffer_m=None):
    """
    Try extraction with primary buffer.
    If results are mostly null and fallback is enabled, retry with a larger buffer.
    """
    geom_primary = buffer_geometry_meters(geom, primary_buffer_m)
    ee_geom_primary = shapely_to_ee_geometry(geom_primary)

    df_primary = imagecollection_to_zone_timeseries(
        ic, ee_geom_primary, scale, dataset_name, start_date, end_date, chunk_years=CHUNK_YEARS
    )
    nn_primary = count_non_null_values(df_primary)
    print(f"  {dataset_name} non-null values (primary buffer {primary_buffer_m} m): {nn_primary}")

    if nn_primary > 0 or not USE_FALLBACK_BUFFER or fallback_buffer_m is None:
        return df_primary

    print(f"  {dataset_name}: retrying with fallback buffer {fallback_buffer_m} m")
    geom_fallback = buffer_geometry_meters(geom, fallback_buffer_m)
    ee_geom_fallback = shapely_to_ee_geometry(geom_fallback)

    df_fallback = imagecollection_to_zone_timeseries(
        ic, ee_geom_fallback, scale, dataset_name, start_date, end_date, chunk_years=CHUNK_YEARS
    )
    nn_fallback = count_non_null_values(df_fallback)
    print(f"  {dataset_name} non-null values (fallback buffer {fallback_buffer_m} m): {nn_fallback}")

    return df_fallback if nn_fallback >= nn_primary else df_primary


# =====================================================
# MAIN
# =====================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PER_ZONE_DIR, exist_ok=True)

    init_ee()
    zones_gdf = build_zone_gdf(CLEANED_SHP)

    print("Zones found:")
    print(zones_gdf[["zone_id", "part_id", "source_label"]])

    gridmet_ic = (
        ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
        .filterDate(START_DATE, END_DATE)
        .map(prep_gridmet)
    )

    chirps_ic = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(START_DATE, END_DATE)
        .map(prep_chirps)
    )

    era5_ic = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(START_DATE, END_DATE)
        .map(prep_era5)
    )

    all_rows = []

    for _, row in zones_gdf.iterrows():
        zone_id = int(row["zone_id"])
        part_id = int(row["part_id"])
        source_name = str(row["source_label"])
        geom = row.geometry

        if geom is None or geom.is_empty:
            print(f"Skipping empty geometry for zone {zone_id}, part {part_id}")
            continue

        print(f"\nProcessing zone {zone_id}, part {part_id}, label={source_name}")

        try:
            df_gridmet = try_dataset_with_buffers(
                gridmet_ic, geom, GRIDMET_SCALE, "GRIDMET",
                START_DATE, END_DATE,
                primary_buffer_m=GRIDMET_BUFFER_M,
                fallback_buffer_m=GRIDMET_BUFFER_M_FALLBACK
            )
            print(f"  GRIDMET rows total: {len(df_gridmet)}")
        except Exception as e:
            print(f"  GRIDMET failed: {e}")
            df_gridmet = pd.DataFrame()

        try:
            df_chirps = try_dataset_with_buffers(
                chirps_ic, geom, CHIRPS_SCALE, "CHIRPS",
                START_DATE, END_DATE,
                primary_buffer_m=CHIRPS_BUFFER_M,
                fallback_buffer_m=CHIRPS_BUFFER_M_FALLBACK
            )
            print(f"  CHIRPS rows total: {len(df_chirps)}")
        except Exception as e:
            print(f"  CHIRPS failed: {e}")
            df_chirps = pd.DataFrame()

        try:
            df_era5 = try_dataset_with_buffers(
                era5_ic, geom, ERA5_SCALE, "ERA5L",
                START_DATE, END_DATE,
                primary_buffer_m=ERA5_BUFFER_M,
                fallback_buffer_m=ERA5_BUFFER_M_FALLBACK
            )
            print(f"  ERA5-Land rows total: {len(df_era5)}")
        except Exception as e:
            print(f"  ERA5-Land failed: {e}")
            print("  If this is a band-name mismatch, send me the traceback and I will patch it.")
            df_era5 = pd.DataFrame()

        merged = merge_on_date([df_gridmet, df_chirps, df_era5])

        if merged.empty:
            print("  No output for this zone.")
            continue

        # Convert numeric-looking columns safely
        for col in merged.columns:
            if col in ["date", "dataset", "source_name"]:
                continue
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

        merged["zone_id"] = zone_id
        merged["part_id"] = part_id
        merged["source_name"] = source_name

        # Wind speed from u/v if available
        if "era5l_u10_ms" in merged.columns and "era5l_v10_ms" in merged.columns:
            u = pd.to_numeric(merged["era5l_u10_ms"], errors="coerce")
            v = pd.to_numeric(merged["era5l_v10_ms"], errors="coerce")
            merged["era5l_windspeed_ms"] = np.sqrt(
                u.to_numpy(dtype=float) ** 2 + v.to_numpy(dtype=float) ** 2
            )

        # Simple RH estimate from t2m and d2m if both exist
        if "era5l_t2m_C" in merged.columns and "era5l_d2m_C" in merged.columns:
            t = pd.to_numeric(merged["era5l_t2m_C"], errors="coerce").to_numpy(dtype=float)
            td = pd.to_numeric(merged["era5l_d2m_C"], errors="coerce").to_numpy(dtype=float)

            es_td = np.exp((17.625 * td) / (243.04 + td))
            es_t = np.exp((17.625 * t) / (243.04 + t))
            rh = 100.0 * (es_td / es_t)
            merged["era5l_rh_pct_est"] = np.clip(rh, 0, 100)

        merged = merged.sort_values("date")

        # Print quick completeness summary
        key_cols = [
            "gridmet_pr_mm", "gridmet_eto_mm", "gridmet_vpd_kPa",
            "chirps_precip_mm",
            "era5l_t2m_C", "era5l_total_precip_mm", "era5l_vswc_l1_m3m3"
        ]
        print("  Non-null summary:")
        for col in key_cols:
            if col in merged.columns:
                n_ok = int(pd.to_numeric(merged[col], errors="coerce").notna().sum())
                print(f"    {col}: {n_ok}")

        safe_name = make_safe_name(source_name)
        out_csv = os.path.join(
            PER_ZONE_DIR,
            f"zone_{zone_id:02d}_part_{part_id}_{safe_name}_weather_timeseries.csv"
        )
        merged.to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv}")

        all_rows.append(merged)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined = combined.sort_values(["zone_id", "part_id", "date"])

        combined_csv = os.path.join(OUTPUT_DIR, "all_zones_daily_weather_timeseries.csv")
        combined.to_csv(combined_csv, index=False)

        print("\nSaved combined CSV:")
        print(combined_csv)
    else:
        print("\nNo combined output created.")


if __name__ == "__main__":
    main()
