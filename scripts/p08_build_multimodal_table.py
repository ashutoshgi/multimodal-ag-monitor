#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:19:36 2026

@author: ashutosh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a multimodal merged table by zone and nearest date.

Merges:
- Sentinel-2 cleaned NDVI
- Landsat cleaned NDVI
- Sentinel-1 cleaned SAR
- Weather daily time series

Output:
- one combined multimodal CSV

"""

import os
import pandas as pd

from eo_config import OUTPUT_DIR, FIELD_NAME, EE_PROJECT, cleaned_shapefile_path, START_YEAR, END_YEAR, WEATHER_START_YEAR, WEATHER_END_YEAR, GRIDMET_BUFFER_M, CHIRPS_BUFFER_M, ERA5_BUFFER_M, LANDSAT_START_YEAR, LANDSAT_END_YEAR, LANDSAT_DOWNLOAD_IMAGES

# =====================================================
# USER SETTINGS
# =====================================================

BASE = OUTPUT_DIR


S2_CSV = os.path.join(
    BASE, "Sentinel2", "per_polygon_time_series", "cleaned_per_polygon_csv",
    "all_polygons_NDVI_cleaned_C_QC.csv"
)

LANDSAT_CSV = os.path.join(
    BASE, "Landsat", "per_polygon_time_series", "cleaned_per_polygon_csv",
    "all_polygons_landsat_NDVI_cleaned_C_QC.csv"
)

S1_CSV = os.path.join(
    BASE, "Sentinel1_TimeSeries", "cleaned_per_zone_csv",
    "all_zones_sentinel1_cleaned.csv"
)

WEATHER_CSV = os.path.join(
    BASE, "Weather_TimeSeries",
    "all_zones_daily_weather_timeseries.csv"
)


# S2_CSV = os.path.join(
#     BASE, "Sentinel2", "per_polygon_time_series", "cleaned_per_polygon_csv",
#     "all_polygons_NDVI_cleaned_C_QC.csv"
# )

# LANDSAT_CSV = os.path.join(
#     BASE, "Landsat", "per_polygon_time_series", "cleaned_per_polygon_csv",
#     "all_polygons_landsat_NDVI_cleaned_C_QC.csv"
# )

# S1_CSV = os.path.join(
#     BASE, "Sentinel1_TimeSeries", "cleaned_per_zone_csv",
#     "all_zones_sentinel1_cleaned.csv"
# )

# WEATHER_CSV = os.path.join(
#     BASE, "Weather_TimeSeries",
#     "all_zones_daily_weather_timeseries.csv"
# )

OUTPUT_DIR = os.path.join(BASE, "Multimodal")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "all_zones_multimodal_merged.csv")

# nearest-date tolerances
S1_TOLERANCE_DAYS = 6
LANDSAT_TOLERANCE_DAYS = 20
WEATHER_TOLERANCE_DAYS = 2

# =====================================================
# HELPERS
# =====================================================
def prep_s2(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    cols = ["zone_id", "part_id", "source_name", "date"]
    for c in ["NDVI_clean", "NDVI_original_obs", "QC_flag"]:
        if c in df.columns:
            cols.append(c)
    df = df[cols].copy()

    rename = {}
    if "NDVI_clean" in df.columns:
        rename["NDVI_clean"] = "s2_ndvi"
    if "NDVI_original_obs" in df.columns:
        rename["NDVI_original_obs"] = "s2_ndvi_obs"
    if "QC_flag" in df.columns:
        rename["QC_flag"] = "s2_qc_flag"

    return df.rename(columns=rename).sort_values(["zone_id", "part_id", "date"])


def prep_landsat(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    cols = ["zone_id", "part_id", "source_name", "date"]
    for c in ["NDVI_clean", "NDVI_original_obs", "QC_flag"]:
        if c in df.columns:
            cols.append(c)
    df = df[cols].copy()

    rename = {}
    if "NDVI_clean" in df.columns:
        rename["NDVI_clean"] = "landsat_ndvi"
    if "NDVI_original_obs" in df.columns:
        rename["NDVI_original_obs"] = "landsat_ndvi_obs"
    if "QC_flag" in df.columns:
        rename["QC_flag"] = "landsat_qc_flag"

    return df.rename(columns=rename).sort_values(["zone_id", "part_id", "date"])


def prep_s1(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    keep = ["zone_id", "part_id", "source_name", "date"]
    for c in [
        "VV_dB_clean", "VH_dB_clean", "VV_minus_VH_dB_clean",
        "VV_plus_VH_mean_dB_clean", "VV_VH_ratio_linear_clean",
        "orbit_pass"
    ]:
        if c in df.columns:
            keep.append(c)

    # fallback if *_clean columns missing
    for rawc in ["VV_dB", "VH_dB", "VV_minus_VH_dB", "VV_plus_VH_mean_dB", "VV_VH_ratio_linear"]:
        if rawc in df.columns and rawc not in keep and f"{rawc}_clean" not in df.columns:
            keep.append(rawc)

    df = df[keep].copy()

    rename = {
        "VV_dB_clean": "s1_vv_db",
        "VH_dB_clean": "s1_vh_db",
        "VV_minus_VH_dB_clean": "s1_vv_minus_vh_db",
        "VV_plus_VH_mean_dB_clean": "s1_vv_plus_vh_mean_db",
        "VV_VH_ratio_linear_clean": "s1_vv_vh_ratio",
        "VV_dB": "s1_vv_db",
        "VH_dB": "s1_vh_db",
        "VV_minus_VH_dB": "s1_vv_minus_vh_db",
        "VV_plus_VH_mean_dB": "s1_vv_plus_vh_mean_db",
        "VV_VH_ratio_linear": "s1_vv_vh_ratio"
    }
    return df.rename(columns=rename).sort_values(["zone_id", "part_id", "date"])


def prep_weather(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    keep = ["zone_id", "part_id", "source_name", "date"]
    useful = [
        "gridmet_pr_mm", "gridmet_tmmn_C", "gridmet_tmmx_C", "gridmet_eto_mm",
        "gridmet_vpd_kPa", "gridmet_wind_ms",
        "chirps_precip_mm",
        "era5l_t2m_C", "era5l_total_precip_mm", "era5l_vswc_l1_m3m3",
        "era5l_vswc_l2_m3m3", "era5l_rh_pct_est", "era5l_windspeed_ms"
    ]
    for c in useful:
        if c in df.columns:
            keep.append(c)

    return df[keep].sort_values(["zone_id", "part_id", "date"]).copy()


def merge_by_zone_and_nearest(left, right, tolerance_days):
    out = []

    left_groups = left.groupby(["zone_id", "part_id"], dropna=False)
    right_groups = {
        (z, p): g.sort_values("date")
        for (z, p), g in right.groupby(["zone_id", "part_id"], dropna=False)
    }

    for key, ldf in left_groups:
        rdf = right_groups.get(key)
        if rdf is None or rdf.empty:
            out.append(ldf.copy())
            continue

        ldf2 = ldf.sort_values("date").copy()
        rdf2 = rdf.sort_values("date").copy()

        ldf2 = ldf2.drop(columns=["source_name"], errors="ignore")
        rdf2 = rdf2.drop(columns=["zone_id", "part_id"], errors="ignore")

        merged = pd.merge_asof(
            ldf2,
            rdf2,
            on="date",
            direction="nearest",
            tolerance=pd.Timedelta(days=tolerance_days),
            suffixes=("", "_r")
        )

        merged["zone_id"] = key[0]
        merged["part_id"] = key[1]

        if "source_name" not in merged.columns:
            if "source_name_r" in merged.columns:
                merged["source_name"] = merged["source_name_r"]
            elif "source_name" in ldf.columns:
                merged["source_name"] = ldf["source_name"].iloc[0]

        drop_cols = [c for c in merged.columns if c.endswith("_r")]
        merged = merged.drop(columns=drop_cols, errors="ignore")

        out.append(merged)

    return pd.concat(out, ignore_index=True).sort_values(["zone_id", "part_id", "date"])


# =====================================================
# MAIN
# =====================================================
if not os.path.exists(S2_CSV):
    raise FileNotFoundError(f"Sentinel-2 cleaned CSV not found:\n{S2_CSV}")
if not os.path.exists(S1_CSV):
    raise FileNotFoundError(f"Sentinel-1 cleaned CSV not found:\n{S1_CSV}")
if not os.path.exists(WEATHER_CSV):
    raise FileNotFoundError(f"Weather CSV not found:\n{WEATHER_CSV}")

print("Sentinel-2 CSV:", S2_CSV)
print("Landsat CSV   :", LANDSAT_CSV)
print("Sentinel-1 CSV:", S1_CSV)
print("Weather CSV   :", WEATHER_CSV)

s2 = prep_s2(S2_CSV)
s1 = prep_s1(S1_CSV)
weather = prep_weather(WEATHER_CSV)

base = s2.copy()

if os.path.exists(LANDSAT_CSV):
    landsat = prep_landsat(LANDSAT_CSV)
    base = merge_by_zone_and_nearest(base, landsat, LANDSAT_TOLERANCE_DAYS)
    print("Merged Landsat.")
else:
    print("Landsat cleaned CSV not found. Continuing without Landsat.")

base = merge_by_zone_and_nearest(base, s1, S1_TOLERANCE_DAYS)
print("Merged Sentinel-1.")

base = merge_by_zone_and_nearest(base, weather, WEATHER_TOLERANCE_DAYS)
print("Merged weather.")

base = base.sort_values(["zone_id", "part_id", "date"])
base.to_csv(OUTPUT_CSV, index=False)

print("\nSaved multimodal table:")
print(OUTPUT_CSV)
print(f"Rows: {len(base)}")
print(f"Columns: {len(base.columns)}")
