#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 2026

Apply NDVI-based outlier removal to all Sentinel-2 indices.

Input:
    all_polygons_indices_time_series.csv

Expected columns:
    zone_id, source_name, part_id, date, NDVI, SAVI, NDWI, NDRE

Outputs:
    - one cleaned CSV per polygon with aligned cleaned rows for all indices
    - one combined cleaned CSV for all polygons

Logic:
    1. Use NDVI only to decide which rows/dates are outliers.
    2. Keep the same retained dates for SAVI, NDWI, and NDRE.
    3. Preserve all index values on those retained dates.

Hare Krishna
"""
#Hare Krishna

import os
import numpy as np
import pandas as pd

from eo_config import OUTPUT_DIR, FIELD_NAME, EE_PROJECT, cleaned_shapefile_path, START_YEAR, END_YEAR, WEATHER_START_YEAR, WEATHER_END_YEAR, GRIDMET_BUFFER_M, CHIRPS_BUFFER_M, ERA5_BUFFER_M, LANDSAT_START_YEAR, LANDSAT_END_YEAR, LANDSAT_DOWNLOAD_IMAGES

# =====================================================
# SETTINGS
# =====================================================

#Hare Krishna
#Hare Krishna
#Hare Krishna
#Hare Krishna

base_folder = os.path.join(OUTPUT_DIR, "Sentinel2", "per_polygon_time_series")
input_csv = os.path.join(base_folder, "all_polygons_indices_time_series.csv")

output_folder = os.path.join(base_folder, "cleaned_per_polygon_csv")
os.makedirs(output_folder, exist_ok=True)

# input_csv = os.path.join(base_folder, "all_polygons_indices_time_series.csv")

# output_folder = os.path.join(base_folder, "cleaned_per_polygon_csv")
# os.makedirs(output_folder, exist_ok=True)


# input_csv = os.path.join(base_folder, "all_polygons_indices_time_series.csv")

# output_folder = os.path.join(base_folder, "cleaned_per_polygon_csv")
# os.makedirs(output_folder, exist_ok=True)

date_col = "date"
ndvi_col = "NDVI"
other_index_cols = ["SAVI", "NDWI", "NDRE"]
all_index_cols = [ndvi_col] + other_index_cols

group_cols = ["zone_id", "source_name", "part_id"]

# Seasonal minimum NDVI thresholds
near_zero_early = 0.05
near_zero_late = 0.20
late_season_fraction = 0.60  # last 40% of each year's observed period treated as late season

# Rolling-window dip removal parameters (NDVI only)
dip_window_days = 21
dip_k = 3.0
dip_min_drop = 0.10

# Final smoothing (NDVI only)
smooth_days = 15

# Optional gentle IQR guardrail after smoothing (NDVI only)
use_iqr_guardrail = True
iqr_k = 2.0

# =====================================================
def compute_season_fraction(dates: pd.Series) -> pd.Series:
    """Return 0..1 position within each calendar year for each date."""
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
                     window_days: int = 21,
                     k: float = 3.0,
                     min_drop: float = 0.10) -> pd.Series:
    """
    Robust rolling-window dip detector.
    Flags points that fall below rolling median by more than max(min_drop, k * robust_sigma),
    where robust_sigma = 1.4826 * rolling_MAD.
    """
    s = series.copy()

    med = s.rolling(f"{window_days}D", center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(f"{window_days}D", center=True, min_periods=3).median()
    robust_sigma = 1.4826 * mad

    thresh = np.maximum(min_drop, k * robust_sigma)
    mask = s < (med - thresh)

    return mask.fillna(False)


def ultrasmooth_optionC_with_qc(df: pd.DataFrame,
                                date_col: str,
                                ndvi_col: str,
                                dip_window_days: int,
                                dip_k: float,
                                dip_min_drop: float,
                                smooth_days: int) -> pd.DataFrame:
    """
    Option C with QC flags:
    - resample to daily
    - interpolate gaps for rolling stats
    - flag cloud/shadow dips
    - remove dips and interpolate through
    - rolling-median smooth
    Returns:
        date, NDVI_clean_daily, QC_flag_daily
    """
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
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(text))


# =====================================================
# MAIN
# =====================================================
if not os.path.exists(input_csv):
    raise FileNotFoundError(f"Input CSV not found:\n{input_csv}")

df = pd.read_csv(input_csv)

required_cols = group_cols + [date_col, ndvi_col]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

for col in all_index_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.floor("D")
df = df.dropna(subset=[date_col, ndvi_col]).copy()

all_cleaned = []

for keys, g_all in df.groupby(group_cols, dropna=False):
    zone_id, source_name, part_id = keys
    print(f"\nProcessing zone_id={zone_id}, source_name={source_name}, part_id={part_id}")

    g_all = g_all.copy().sort_values(date_col)
    original_rows = len(g_all)

    # -------------------------------------------------
    # STEP 1: Remove duplicate dates (keep first only)
    # -------------------------------------------------
    g_all = g_all.drop_duplicates(subset=[date_col], keep="first").copy()

    # Preserve original first-per-date values for all indices
    for col in all_index_cols:
        if col in g_all.columns:
            g_all[f"{col}_raw_first_per_date"] = g_all[col].astype(float)
            g_all[f"{col}_original_obs"] = g_all[col].astype(float)

    # Work on NDVI-only frame for outlier logic
    g_ndvi = g_all[[date_col, ndvi_col, "NDVI_raw_first_per_date", "NDVI_original_obs"]].copy()

    # -------------------------------------------------
    # STEP 2: Season-aware minimum NDVI filter
    # -------------------------------------------------
    season_frac = compute_season_fraction(g_ndvi[date_col])
    is_late = season_frac >= late_season_fraction
    thresh = np.where(is_late, near_zero_late, near_zero_early)

    g_ndvi = g_ndvi[g_ndvi[ndvi_col] >= thresh].copy()

    if len(g_ndvi) < 3:
        kept_dates = set(g_ndvi[date_col].drop_duplicates())
        out = g_all[g_all[date_col].isin(kept_dates)].copy()

        out["NDVI_clean"] = out["NDVI_original_obs"]
        out["QC_flag"] = 0

        keep_cols = group_cols + [date_col]
        for col in all_index_cols:
            if f"{col}_raw_first_per_date" in out.columns:
                keep_cols.extend([f"{col}_raw_first_per_date", f"{col}_original_obs"])
        keep_cols.extend(["NDVI_clean", "QC_flag"])

        out = out[keep_cols].sort_values(date_col)

        safe_name = make_safe_name(source_name)
        out_csv = os.path.join(output_folder, f"zone_{int(zone_id):02d}_{safe_name}_part_{int(part_id)}_all_indices_cleaned_by_NDVI.csv")
        out.to_csv(out_csv, index=False)
        print(f"  Saved (few points): {out_csv}")

        all_cleaned.append(out)
        continue

    # -------------------------------------------------
    # STEP 3: NDVI ultrasmooth with rolling dip removal + QC
    # -------------------------------------------------
    daily_clean = ultrasmooth_optionC_with_qc(
        g_ndvi, date_col, ndvi_col,
        dip_window_days=dip_window_days,
        dip_k=dip_k,
        dip_min_drop=dip_min_drop,
        smooth_days=smooth_days
    )

    obs_dates = g_ndvi[date_col].drop_duplicates()
    sampled = daily_clean[daily_clean[date_col].isin(obs_dates)].copy()

    merge_map = g_ndvi[[date_col, "NDVI_raw_first_per_date", "NDVI_original_obs"]].drop_duplicates(subset=[date_col], keep="first")
    sampled = sampled.merge(merge_map, on=date_col, how="left")
    sampled = sampled.rename(columns={
        "NDVI_clean_daily": "NDVI_clean",
        "QC_flag_daily": "QC_flag"
    })

    # -------------------------------------------------
    # STEP 4: Optional IQR guardrail after smoothing
    # -------------------------------------------------
    if use_iqr_guardrail and len(sampled) >= 8:
        lower, upper = iqr_bounds(sampled["NDVI_clean"], k=iqr_k)
        sampled = sampled[(sampled["NDVI_clean"] >= lower) & (sampled["NDVI_clean"] <= upper)].copy()

    # Final kept dates = NDVI-cleaned dates
    kept_dates = set(sampled[date_col].drop_duplicates())

    # Apply same retained dates to all indices
    out = g_all[g_all[date_col].isin(kept_dates)].copy()
    out = out.merge(sampled[[date_col, "NDVI_clean", "QC_flag"]], on=date_col, how="left")

    out["zone_id"] = zone_id
    out["source_name"] = source_name
    out["part_id"] = part_id

    keep_cols = group_cols + [date_col]
    for col in all_index_cols:
        if f"{col}_raw_first_per_date" in out.columns:
            keep_cols.extend([f"{col}_raw_first_per_date", f"{col}_original_obs"])
    keep_cols.extend(["NDVI_clean", "QC_flag"])

    out = out[keep_cols].sort_values(date_col)

    safe_name = make_safe_name(source_name)
    out_csv = os.path.join(output_folder, f"zone_{int(zone_id):02d}_{safe_name}_part_{int(part_id)}_all_indices_cleaned_by_NDVI.csv")
    out.to_csv(out_csv, index=False)

    print(f"  Saved: {out_csv}")
    print(f"  Rows: {original_rows} -> {len(out)}")
    print(f"  QC corrected points (flag=1): {int(sampled['QC_flag'].sum())}")

    all_cleaned.append(out)

# Save combined cleaned CSV
if all_cleaned:
    combined = pd.concat(all_cleaned, ignore_index=True)
    combined = combined.sort_values(["zone_id", "part_id", date_col])

    combined_csv = os.path.join(output_folder, "all_polygons_all_indices_cleaned_by_NDVI.csv")
    combined.to_csv(combined_csv, index=False)

    print("\nDone. Combined cleaned CSV saved to:")
    print(combined_csv)
else:
    print("\nNo cleaned outputs were created.")
