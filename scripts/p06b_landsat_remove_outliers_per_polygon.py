#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove Landsat NDVI outliers from the per-polygon combined CSV.
Adapted to the Landsat downloader output.

Input:
    all_polygons_landsat_ndvi.csv
Output:
    cleaned per-polygon CSVs + combined cleaned CSV


"""

import os
import numpy as np
import pandas as pd

from eo_config import OUTPUT_DIR, FIELD_NAME, EE_PROJECT, cleaned_shapefile_path, START_YEAR, END_YEAR, WEATHER_START_YEAR, WEATHER_END_YEAR, GRIDMET_BUFFER_M, CHIRPS_BUFFER_M, ERA5_BUFFER_M, LANDSAT_START_YEAR, LANDSAT_END_YEAR, LANDSAT_DOWNLOAD_IMAGES

# =====================================================
# SETTINGS
# =====================================================

base_folder = os.path.join(OUTPUT_DIR, "Landsat", "per_polygon_time_series")
input_csv = os.path.join(base_folder, "all_polygons_landsat_ndvi.csv")

output_folder = os.path.join(base_folder, "cleaned_per_polygon_csv")
os.makedirs(output_folder, exist_ok=True)


# input_csv = os.path.join(base_folder, "all_polygons_landsat_ndvi.csv")
# output_folder = os.path.join(base_folder, "cleaned_per_polygon_csv")

# input_csv = os.path.join(base_folder, "all_polygons_landsat_ndvi.csv")

# output_folder = os.path.join(base_folder, "cleaned_per_polygon_csv")
# os.makedirs(output_folder, exist_ok=True)

# input_csv = os.path.join(base_folder, "all_polygons_landsat_ndvi.csv")

# output_folder = os.path.join(base_folder, "cleaned_per_polygon_csv")
# os.makedirs(output_folder, exist_ok=True)

date_col = "date"
ndvi_col = "NDVI"
group_cols = ["zone_id", "source_name", "part_id"]

near_zero_early = 0.05
near_zero_late = 0.20
late_season_fraction = 0.60

dip_window_days = 45   # longer for Landsat because revisit is sparser than Sentinel-2
dip_k = 3.0
dip_min_drop = 0.10
smooth_days = 45

use_iqr_guardrail = True
iqr_k = 2.0


def compute_season_fraction(dates: pd.Series) -> pd.Series:
    years = dates.dt.year
    frac = pd.Series(index=dates.index, dtype=float)
    for y in years.dropna().unique():
        idx = dates.index[years == y]
        dmin = dates.loc[idx].min()
        dmax = dates.loc[idx].max()
        if pd.isna(dmin) or pd.isna(dmax) or dmin == dmax:
            frac.loc[idx] = 1.0
        else:
            frac.loc[idx] = ((dates.loc[idx] - dmin) / (dmax - dmin)).astype(float)
    return frac


def iqr_bounds(series: pd.Series, k: float = 1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def rolling_dip_mask(series: pd.Series,
                     window_days: int = 45,
                     k: float = 3.0,
                     min_drop: float = 0.10) -> pd.Series:
    med = series.rolling(f"{window_days}D", center=True, min_periods=3).median()
    mad = (series - med).abs().rolling(f"{window_days}D", center=True, min_periods=3).median()
    robust_sigma = 1.4826 * mad
    thresh = np.maximum(min_drop, k * robust_sigma)
    mask = series < (med - thresh)
    return mask.fillna(False)


def ultrasmooth_optionC_with_qc(df: pd.DataFrame,
                                date_col: str,
                                ndvi_col: str,
                                dip_window_days: int,
                                dip_k: float,
                                dip_min_drop: float,
                                smooth_days: int) -> pd.DataFrame:
    d = df[[date_col, ndvi_col]].copy().sort_values(date_col)
    d = d.dropna(subset=[date_col, ndvi_col]).set_index(date_col)

    daily = d.resample("D").mean()
    daily[ndvi_col] = daily[ndvi_col].interpolate(method="time", limit_direction="both")

    dip_mask = rolling_dip_mask(
        daily[ndvi_col],
        window_days=dip_window_days,
        k=dip_k,
        min_drop=dip_min_drop
    )

    daily.loc[dip_mask, ndvi_col] = np.nan
    daily[ndvi_col] = daily[ndvi_col].interpolate(method="time", limit_direction="both")

    ndvi_clean = (
        daily[ndvi_col]
        .rolling(f"{smooth_days}D", center=True, min_periods=3)
        .median()
        .interpolate(method="time", limit_direction="both")
    )

    out = daily.reset_index()[[date_col]].copy()
    out["NDVI_clean_daily"] = ndvi_clean.values
    out["QC_flag_daily"] = dip_mask.reset_index(drop=True).astype(int).values
    return out


def make_safe_name(text):
    return ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in str(text))


if not os.path.exists(input_csv):
    raise FileNotFoundError(f"Input CSV not found:\n{input_csv}")

df = pd.read_csv(input_csv)
required_cols = group_cols + [date_col, ndvi_col]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.floor("D")
df = df.dropna(subset=[date_col, ndvi_col]).copy()

all_cleaned = []

for keys, g in df.groupby(group_cols, dropna=False):
    zone_id, source_name, part_id = keys
    print(f"\nProcessing zone_id={zone_id}, source_name={source_name}, part_id={part_id}")

    g = g.copy().sort_values(date_col)
    original_rows = len(g)
    g = g.drop_duplicates(subset=[date_col], keep="first").copy()
    g["NDVI_raw_first_per_date"] = g[ndvi_col].astype(float)

    season_frac = compute_season_fraction(g[date_col])
    is_late = season_frac >= late_season_fraction
    thresh = np.where(is_late, near_zero_late, near_zero_early)

    g["NDVI_original_obs"] = g[ndvi_col].astype(float)
    g = g[g[ndvi_col] >= thresh].copy()

    if len(g) < 3:
        out = g[[date_col, "NDVI_raw_first_per_date", "NDVI_original_obs"]].copy()
        out["NDVI_clean"] = out["NDVI_original_obs"]
        out["QC_flag"] = 0
        out["zone_id"] = zone_id
        out["source_name"] = source_name
        out["part_id"] = part_id
        out = out[["zone_id", "source_name", "part_id", date_col,
                   "NDVI_raw_first_per_date", "NDVI_original_obs", "NDVI_clean", "QC_flag"]]

        safe_name = make_safe_name(source_name)
        out_csv = os.path.join(output_folder, f"zone_{int(zone_id):02d}_{safe_name}_part_{int(part_id)}_landsat_cleaned_C_QC.csv")
        out.to_csv(out_csv, index=False)
        print(f"  Saved (few points): {out_csv}")
        all_cleaned.append(out)
        continue

    daily_clean = ultrasmooth_optionC_with_qc(
        g, date_col, ndvi_col,
        dip_window_days=dip_window_days,
        dip_k=dip_k,
        dip_min_drop=dip_min_drop,
        smooth_days=smooth_days
    )

    obs_dates = g[date_col].drop_duplicates()
    sampled = daily_clean[daily_clean[date_col].isin(obs_dates)].copy()

    merge_map = g[[date_col, "NDVI_raw_first_per_date", "NDVI_original_obs"]].drop_duplicates(subset=[date_col], keep="first")
    sampled = sampled.merge(merge_map, on=date_col, how="left")
    sampled = sampled.rename(columns={"NDVI_clean_daily": "NDVI_clean", "QC_flag_daily": "QC_flag"})

    if use_iqr_guardrail and len(sampled) >= 8:
        lower, upper = iqr_bounds(sampled["NDVI_clean"], k=iqr_k)
        sampled = sampled[(sampled["NDVI_clean"] >= lower) & (sampled["NDVI_clean"] <= upper)].copy()

    sampled["zone_id"] = zone_id
    sampled["source_name"] = source_name
    sampled["part_id"] = part_id
    sampled = sampled[["zone_id", "source_name", "part_id", date_col,
                       "NDVI_raw_first_per_date", "NDVI_original_obs", "NDVI_clean", "QC_flag"]].sort_values(date_col)

    safe_name = make_safe_name(source_name)
    out_csv = os.path.join(output_folder, f"zone_{int(zone_id):02d}_{safe_name}_part_{int(part_id)}_landsat_cleaned_C_QC.csv")
    sampled.to_csv(out_csv, index=False)

    print(f"  Saved: {out_csv}")
    print(f"  Rows: {original_rows} -> {len(sampled)}")
    print(f"  QC corrected points (flag=1): {int(sampled['QC_flag'].sum())}")

    all_cleaned.append(sampled)

if all_cleaned:
    combined = pd.concat(all_cleaned, ignore_index=True)
    combined = combined.sort_values(["zone_id", "part_id", date_col])
    combined_csv = os.path.join(output_folder, "all_polygons_landsat_NDVI_cleaned_C_QC.csv")
    combined.to_csv(combined_csv, index=False)
    print("\nDone. Combined cleaned CSV saved to:")
    print(combined_csv)
else:
    print("\nNo cleaned outputs were created.")
