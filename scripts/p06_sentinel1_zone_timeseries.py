#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p06_sentinel1_zone_timeseries_generic_FIXED.py

Extract Sentinel-1 zone-wise SAR time series for {FIELD_NAME}.

This fixed version handles occasional Sentinel-1 images where reduceRegion or
metadata returns null values. The earlier version failed with:
Dictionary.set: Parameter 'value' is required and may not be null.

Outputs:
- one CSV per zone
- one combined CSV for all zones

Indicators saved:
- VV_dB
- VH_dB
- VV_minus_VH_dB
- VH_minus_VV_dB
- VV_VH_ratio_linear
- VH_VV_ratio_linear
- VV_plus_VH_mean_dB
- orbit pass / orbit number / platform

"""

import os
import json
from datetime import datetime

import ee
import geopandas as gpd
import pandas as pd
import numpy as np

from eo_config import OUTPUT_DIR, FIELD_NAME, EE_PROJECT, cleaned_shapefile_path, START_YEAR, END_YEAR, WEATHER_START_YEAR, WEATHER_END_YEAR, GRIDMET_BUFFER_M, CHIRPS_BUFFER_M, ERA5_BUFFER_M, LANDSAT_START_YEAR, LANDSAT_END_YEAR, LANDSAT_DOWNLOAD_IMAGES, S1_ORBIT

# =====================================================
# USER SETTINGS
# =====================================================
BASE_DIR = OUTPUT_DIR
CLEANED_SHP = cleaned_shapefile_path()

OUTPUT_DIR = os.path.join(BASE_DIR, "Sentinel1_TimeSeries")
PER_ZONE_DIR = os.path.join(OUTPUT_DIR, "per_zone_csv")

# EE_PROJECT loaded from config

# Sentinel-1 stable archive period.
START_DATE = "2014-10-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Pull year-by-year for safety.
CHUNK_YEARS = 1

# Filtering.
ORBIT_PASS = S1_ORBIT   # None, 'ASCENDING', or 'DESCENDING'
SCALE = 10

# Sentinel value used inside Earth Engine because null properties cannot be set.
NO_DATA_VALUE = -9999.0

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
            axis=1,
        )
    else:
        gdf["source_label"] = gdf[source_col].astype(str)

    return gdf


def build_s1_collection(aoi, start_date, end_date, orbit_pass=None):
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(start_date, end_date)
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("resolution_meters", 10))
    )

    if orbit_pass in ["ASCENDING", "DESCENDING"]:
        s1 = s1.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))

    # Keep properties by selecting bands after filters only.
    return s1.select(["VV", "VH"])


def safe_img_prop(img, prop_name, default_value):
    """Return image property if present; otherwise a non-null default."""
    return ee.Algorithms.If(img.propertyNames().contains(prop_name), img.get(prop_name), default_value)


def collection_to_timeseries(ic, ee_geom, start_date, end_date, chunk_years=1):
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
            img = ee.Image(img)
            date = img.date().format("YYYY-MM-dd")
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee_geom,
                scale=SCALE,
                maxPixels=1e13,
                bestEffort=True,
            )

            # Do not set null values in Earth Engine. Nulls can crash Dictionary.set.
            vv_val = ee.Algorithms.If(stats.contains("VV"), stats.get("VV"), NO_DATA_VALUE)
            vh_val = ee.Algorithms.If(stats.contains("VH"), stats.get("VH"), NO_DATA_VALUE)

            props = ee.Dictionary({
                "date": date,
                "VV": vv_val,
                "VH": vh_val,
                "has_VV": stats.contains("VV"),
                "has_VH": stats.contains("VH"),
                "orbit_pass": safe_img_prop(img, "orbitProperties_pass", "UNKNOWN"),
                "relative_orbit": safe_img_prop(img, "relativeOrbitNumber_start", -9999),
                "platform": safe_img_prop(img, "platform_number", "UNKNOWN"),
                "slice_number": safe_img_prop(img, "sliceNumber", -9999),
                "system_index": safe_img_prop(img, "system:index", "UNKNOWN"),
            })
            return ee.Feature(None, props)

        try:
            fc = ee.FeatureCollection(ic_chunk.map(per_image))
            info = fc.getInfo()
            rows = [feat["properties"] for feat in info.get("features", [])]
        except Exception as e:
            print(f"    Sentinel-1 chunk failed for {chunk_start} to {chunk_end}: {e}")
            rows = []

        if rows:
            all_rows.extend(rows)

        print(f"    Sentinel-1: pulled {len(rows)} rows for {chunk_start} to {chunk_end}")

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Convert sentinel values back to NaN in Python.
    for col in ["VV", "VH", "relative_orbit", "slice_number"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] == NO_DATA_VALUE, col] = np.nan

    # Drop S1 images where zone mean could not be computed.
    before = len(df)
    df = df.dropna(subset=["VV", "VH"], how="any").copy()
    dropped = before - len(df)
    if dropped:
        print(f"    Dropped {dropped} rows with missing VV/VH after extraction.")

    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)

    return df


def add_s1_indicators(df):
    if df.empty:
        return df

    for col in ["VV", "VH"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns={"VV": "VV_dB", "VH": "VH_dB"})

    if "VV_dB" in df.columns and "VH_dB" in df.columns:
        vv = df["VV_dB"].to_numpy(dtype=float)
        vh = df["VH_dB"].to_numpy(dtype=float)

        vv_lin = 10 ** (vv / 10.0)
        vh_lin = 10 ** (vh / 10.0)

        df["VV_minus_VH_dB"] = vv - vh
        df["VH_minus_VV_dB"] = vh - vv
        df["VV_plus_VH_mean_dB"] = (vv + vh) / 2.0

        vv_vh = np.divide(
            vv_lin, vh_lin,
            out=np.full_like(vv_lin, np.nan),
            where=np.isfinite(vv_lin) & np.isfinite(vh_lin) & (vh_lin != 0),
        )
        vh_vv = np.divide(
            vh_lin, vv_lin,
            out=np.full_like(vh_lin, np.nan),
            where=np.isfinite(vv_lin) & np.isfinite(vh_lin) & (vv_lin != 0),
        )
        df["VV_VH_ratio_linear"] = vv_vh
        df["VH_VV_ratio_linear"] = vh_vv

    return df


def add_orbitwise_anomalies(df):
    """
    Add simple orbit-specific rolling anomalies for moisture-sensitive interpretation.
    This helps because ascending and descending passes can differ systematically.
    """
    if df.empty or "orbit_pass" not in df.columns:
        return df

    df = df.sort_values(["orbit_pass", "date"]).copy()

    for base_col in ["VV_dB", "VH_dB", "VV_minus_VH_dB", "VH_minus_VV_dB"]:
        if base_col not in df.columns:
            continue

        anom_col = f"{base_col}_anom_5obs"
        df[anom_col] = (
            df.groupby("orbit_pass")[base_col]
              .transform(lambda s: s - s.rolling(window=5, min_periods=3, center=True).median())
        )

    return df


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

        ee_geom = shapely_to_ee_geometry(geom)
        s1_ic = build_s1_collection(ee_geom, START_DATE, END_DATE, orbit_pass=ORBIT_PASS)

        df = collection_to_timeseries(
            s1_ic,
            ee_geom,
            START_DATE,
            END_DATE,
            chunk_years=CHUNK_YEARS,
        )
        print(f"  Sentinel-1 rows total after dropping missing VV/VH: {len(df)}")

        if df.empty:
            print("  No output for this zone.")
            continue

        df = add_s1_indicators(df)
        df = add_orbitwise_anomalies(df)

        df["zone_id"] = zone_id
        df["part_id"] = part_id
        df["source_name"] = source_name

        keep_cols = [
            "date", "orbit_pass", "relative_orbit", "platform", "slice_number", "system_index",
            "VV_dB", "VH_dB",
            "VV_minus_VH_dB", "VH_minus_VV_dB", "VV_plus_VH_mean_dB",
            "VV_VH_ratio_linear", "VH_VV_ratio_linear",
            "VV_dB_anom_5obs", "VH_dB_anom_5obs",
            "VV_minus_VH_dB_anom_5obs", "VH_minus_VV_dB_anom_5obs",
            "zone_id", "part_id", "source_name",
        ]
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols].sort_values("date")

        safe_name = make_safe_name(source_name)
        out_csv = os.path.join(
            PER_ZONE_DIR,
            f"zone_{zone_id:02d}_part_{part_id}_{safe_name}_sentinel1_timeseries.csv",
        )
        df.to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv}")

        all_rows.append(df)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined = combined.sort_values(["zone_id", "part_id", "date"])
        combined_csv = os.path.join(OUTPUT_DIR, "all_zones_sentinel1_timeseries.csv")
        combined.to_csv(combined_csv, index=False)

        print("\nSaved combined CSV:")
        print(combined_csv)
        print("\nSummary:")
        print(combined.groupby([combined["date"].dt.year.rename("year"), "source_name"]).agg(
            n_rows=("date", "size"),
            n_dates=("date", "nunique"),
            vv_mean=("VV_dB", "mean"),
            vh_mean=("VH_dB", "mean"),
        ))
    else:
        print("\nNo combined output created.")


if __name__ == "__main__":
    main()
