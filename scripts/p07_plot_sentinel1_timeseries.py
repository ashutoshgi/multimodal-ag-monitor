#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p07 Sentinel-1 cleaning and plotting for {FIELD_NAME}.

Purpose
-------
Clean Sentinel-1 zone-wise time series for crop-growth analysis.
This version is designed for denser Sentinel-1 downloads, e.g., checking every
5 days in p01 and then extracting whatever Sentinel-1 observations are available.

Input
-----
    all_zones_sentinel1_timeseries.csv

Expected important columns
--------------------------
    zone_id, part_id, source_name, date,
    VV_dB, VH_dB,
    optional: VV_minus_VH_dB, VV_plus_VH_mean_dB,
              VV_VH_ratio_linear, orbit_pass, relative_orbit, platform

Outputs
-------
    cleaned_per_zone_csv/
        - one cleaned CSV per zone/part
        - all_zones_sentinel1_cleaned.csv
        - sentinel1_cleaning_summary.csv
        - plots_yearwise_individual/*.png

Main improvements over previous p07
-----------------------------------
1. Keeps raw values, despiked values, smoothed values, and QC flags.
2. Uses robust rolling-median/MAD despiking for SAR time-series noise.
3. Smooths the despiked signal using a short rolling median window.
4. Adds VH/VV linear ratio and VH - VV dB ratio-like indicator.
5. Keeps VV/VH columns too, so downstream scripts do not break.
6. Optionally cleans within orbit/pass when enough observations are available.

Notes
-----
- For crop growth, VH and VH/VV are often useful because VH is sensitive to
  volume scattering from canopy structure, while VV is more influenced by
  surface scattering and soil/canopy geometry.
- Do not make the smoothing window too large if you want to preserve crop
  phenology changes after rainfall or irrigation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eo_config import OUTPUT_DIR, FIELD_NAME, EE_PROJECT, cleaned_shapefile_path, START_YEAR, END_YEAR, WEATHER_START_YEAR, WEATHER_END_YEAR, GRIDMET_BUFFER_M, CHIRPS_BUFFER_M, ERA5_BUFFER_M, LANDSAT_START_YEAR, LANDSAT_END_YEAR, LANDSAT_DOWNLOAD_IMAGES

# =====================================================
# USER SETTINGS
# =====================================================

BASE_DIR = os.path.join(OUTPUT_DIR, "Sentinel1_TimeSeries")
INPUT_CSV = os.path.join(BASE_DIR, "all_zones_sentinel1_timeseries.csv")

CLEAN_DIR = os.path.join(BASE_DIR, "cleaned_per_zone_csv")
PLOT_DIR = os.path.join(CLEAN_DIR, "plots_yearwise_individual")
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

GROUP_COLS = ["zone_id", "part_id", "source_name"]
DATE_COL = "date"

# Cleaning controls
# With 5-day checking in p01, Sentinel-1 observations are still irregular.
# Therefore these windows are observation-count based, not daily-window based.
DESPIKE_WINDOW_OBS = 7       # robust rolling window for outlier detection
SMOOTH_WINDOW_OBS = 5        # light smoothing after despiking
MAD_K = 3.5                  # larger = less aggressive outlier removal
MIN_POINTS_FOR_CLEANING = 6

# If True, clean separate orbit streams when possible.
# This is safer because ascending/descending and relative orbits may have
# systematic backscatter offsets.
CLEAN_WITHIN_ORBIT_IF_POSSIBLE = True
MIN_POINTS_PER_ORBIT_GROUP = 6

# Plot controls
SHOW_RAW_ON_PLOTS = True
SHOW_OUTLIERS_ON_PLOTS = True
FIGSIZE = (12, 5.5)
DPI = 220

# Plot these variables yearwise. VV/VH is no longer plotted by default;
# VH/VV is added for crop-growth interpretation.
PLOT_CONFIGS = {
    "VV_dB": {
        "clean_col": "VV_dB_clean",
        "raw_col": "VV_dB_raw",
        "flag_col": "VV_dB_flag",
        "ylabel": "VV backscatter (dB)",
        "title": "VV backscatter",
        "filename": "VV_dB",
    },
    "VH_dB": {
        "clean_col": "VH_dB_clean",
        "raw_col": "VH_dB_raw",
        "flag_col": "VH_dB_flag",
        "ylabel": "VH backscatter (dB)",
        "title": "VH backscatter",
        "filename": "VH_dB",
    },
    "VH_minus_VV_dB": {
        "clean_col": "VH_minus_VV_dB_clean",
        "raw_col": "VH_minus_VV_dB_raw",
        "flag_col": "VH_minus_VV_dB_flag",
        "ylabel": "VH - VV (dB)",
        "title": "VH minus VV",
        "filename": "VH_minus_VV_dB",
    },
    "VH_VV_ratio_linear": {
        "clean_col": "VH_VV_ratio_linear_clean",
        "raw_col": "VH_VV_ratio_linear_raw",
        "flag_col": "VH_VV_ratio_linear_flag",
        "ylabel": "VH / VV ratio (linear)",
        "title": "VH/VV ratio",
        "filename": "VH_VV_ratio_linear",
    },
}

# Variable-specific minimum changes required before a point is allowed to be
# flagged as an outlier. This prevents the MAD test from over-flagging small
# natural SAR fluctuations.
MIN_ABS_DIFF = {
    "VV_dB": 2.5,
    "VH_dB": 2.5,
    "VV_minus_VH_dB": 1.5,
    "VH_minus_VV_dB": 1.5,
    "VV_plus_VH_mean_dB": 2.0,
    "VV_VH_ratio_linear": 2.0,
    "VH_VV_ratio_linear": 0.12,
}

# Columns to clean if present or created.
S1_COLS = [
    "VV_dB",
    "VH_dB",
    "VV_minus_VH_dB",
    "VH_minus_VV_dB",
    "VV_plus_VH_mean_dB",
    "VV_VH_ratio_linear",
    "VH_VV_ratio_linear",
]


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def make_safe_name(text):
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(text))


def ensure_s1_indicators(df):
    """
    Make sure both VV/VH and VH/VV ratio indicators exist.
    Input VV_dB and VH_dB are assumed to be in dB.
    """
    df = df.copy()

    for col in ["VV_dB", "VH_dB"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "VV_dB" in df.columns and "VH_dB" in df.columns:
        vv = df["VV_dB"].to_numpy(dtype=float)
        vh = df["VH_dB"].to_numpy(dtype=float)

        vv_lin = 10.0 ** (vv / 10.0)
        vh_lin = 10.0 ** (vh / 10.0)

        if "VV_minus_VH_dB" not in df.columns:
            df["VV_minus_VH_dB"] = df["VV_dB"] - df["VH_dB"]

        # Crop-growth-oriented reciprocal difference.
        # This is the dB equivalent of log10(VH/VV).
        df["VH_minus_VV_dB"] = df["VH_dB"] - df["VV_dB"]

        if "VV_plus_VH_mean_dB" not in df.columns:
            df["VV_plus_VH_mean_dB"] = (df["VV_dB"] + df["VH_dB"]) / 2.0

        if "VV_VH_ratio_linear" not in df.columns:
            df["VV_VH_ratio_linear"] = np.divide(
                vv_lin,
                vh_lin,
                out=np.full_like(vv_lin, np.nan, dtype=float),
                where=np.isfinite(vv_lin) & np.isfinite(vh_lin) & (vh_lin != 0),
            )

        # New preferred crop-growth ratio.
        df["VH_VV_ratio_linear"] = np.divide(
            vh_lin,
            vv_lin,
            out=np.full_like(vh_lin, np.nan, dtype=float),
            where=np.isfinite(vh_lin) & np.isfinite(vv_lin) & (vv_lin != 0),
        )

    return df


def robust_despike_and_smooth(series, col_name):
    """
    Robust SAR cleaning for one numeric series.

    Returns
    -------
    raw : original numeric series
    despiked : outliers removed and interpolated
    smooth : despiked series after rolling median smoothing
    flag : 1 for points flagged as outliers, else 0
    """
    raw = pd.to_numeric(series, errors="coerce").copy()

    if raw.notna().sum() < MIN_POINTS_FOR_CLEANING:
        out = raw.interpolate(limit_direction="both")
        return raw, out, out, pd.Series(0, index=raw.index, dtype=int)

    med = raw.rolling(
        window=DESPIKE_WINDOW_OBS,
        center=True,
        min_periods=max(3, min(DESPIKE_WINDOW_OBS, raw.notna().sum()) // 2),
    ).median()

    mad = (raw - med).abs().rolling(
        window=DESPIKE_WINDOW_OBS,
        center=True,
        min_periods=max(3, min(DESPIKE_WINDOW_OBS, raw.notna().sum()) // 2),
    ).median()

    robust_sigma = 1.4826 * mad
    min_abs = MIN_ABS_DIFF.get(col_name, 0.0)
    threshold = np.maximum(MAD_K * robust_sigma, min_abs)

    flag_bool = ((raw - med).abs() > threshold).fillna(False)

    despiked = raw.copy()
    despiked[flag_bool] = np.nan
    despiked = despiked.interpolate(limit_direction="both")

    smooth = despiked.rolling(
        window=SMOOTH_WINDOW_OBS,
        center=True,
        min_periods=2,
    ).median()
    smooth = smooth.interpolate(limit_direction="both")

    return raw, despiked, smooth, flag_bool.astype(int)


def choose_cleaning_subgroups(g):
    """
    Return a list of index arrays to clean. Prefer orbit-specific cleaning if
    enough observations exist; otherwise clean the full group together.
    """
    if not CLEAN_WITHIN_ORBIT_IF_POSSIBLE:
        return [("all", g.index)]

    possible_cols = [c for c in ["orbit_pass", "relative_orbit"] if c in g.columns]
    if not possible_cols:
        return [("all", g.index)]

    subgroups = []
    for key, sub in g.groupby(possible_cols, dropna=False):
        if len(sub) >= MIN_POINTS_PER_ORBIT_GROUP:
            subgroups.append((str(key), sub.index))

    covered = set()
    for _, idx in subgroups:
        covered.update(idx.tolist())

    remaining = [idx for idx in g.index if idx not in covered]
    if len(remaining) >= MIN_POINTS_FOR_CLEANING:
        subgroups.append(("remaining", pd.Index(remaining)))
    elif len(subgroups) == 0:
        subgroups = [("all", g.index)]

    return subgroups


def clean_group(g):
    """
    Clean all S1 variables in one zone/part/source group.
    """
    g = g.sort_values(DATE_COL).copy()

    # Initialize output columns for all variables. Then fill subgroup-wise.
    for col in S1_COLS:
        if col not in g.columns:
            continue
        g[f"{col}_raw"] = pd.to_numeric(g[col], errors="coerce")
        g[f"{col}_despiked"] = np.nan
        g[f"{col}_smooth"] = np.nan
        g[f"{col}_clean"] = np.nan
        g[f"{col}_flag"] = 0

    for subgroup_name, idx in choose_cleaning_subgroups(g):
        sub = g.loc[idx].sort_values(DATE_COL).copy()

        for col in S1_COLS:
            if col not in sub.columns:
                continue
            raw, despiked, smooth, flag = robust_despike_and_smooth(sub[col], col)

            g.loc[sub.index, f"{col}_raw"] = raw.values
            g.loc[sub.index, f"{col}_despiked"] = despiked.values
            g.loc[sub.index, f"{col}_smooth"] = smooth.values

            # Downstream scripts expect *_clean. Here *_clean means smoothed
            # after robust despiking.
            g.loc[sub.index, f"{col}_clean"] = smooth.values
            g.loc[sub.index, f"{col}_flag"] = flag.values

    return g.sort_values(DATE_COL)


def save_individual_plot(d, cfg, base_title, out_png):
    clean_col = cfg["clean_col"]
    raw_col = cfg["raw_col"]
    flag_col = cfg["flag_col"]

    if clean_col not in d.columns or not d[clean_col].notna().any():
        return False

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if SHOW_RAW_ON_PLOTS and raw_col in d.columns and d[raw_col].notna().any():
        ax.plot(
            d[DATE_COL],
            d[raw_col],
            marker="o",
            linestyle="None",
            markersize=4,
            alpha=0.45,
            label="Raw observations",
        )

    ax.plot(
        d[DATE_COL],
        d[clean_col],
        marker="o",
        linewidth=2.1,
        markersize=4,
        label="Despiked + smoothed",
    )

    if SHOW_OUTLIERS_ON_PLOTS and flag_col in d.columns:
        flagged = d[(d[flag_col] == 1) & d[raw_col].notna()].copy() if raw_col in d.columns else pd.DataFrame()
        if not flagged.empty:
            ax.scatter(
                flagged[DATE_COL],
                flagged[raw_col],
                marker="x",
                s=80,
                label="Flagged outlier",
            )

    ax.set_title(f"{base_title} - {cfg['title']}")
    ax.set_xlabel("Date")
    ax.set_ylabel(cfg["ylabel"])
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    return True


def summarize_cleaning(g, zone_id, part_id, source_name):
    rows = []
    for col in S1_COLS:
        flag_col = f"{col}_flag"
        raw_col = f"{col}_raw"
        clean_col = f"{col}_clean"
        if flag_col not in g.columns:
            continue
        rows.append({
            "zone_id": zone_id,
            "part_id": part_id,
            "source_name": source_name,
            "variable": col,
            "n_total": int(len(g)),
            "n_raw_valid": int(g[raw_col].notna().sum()) if raw_col in g.columns else 0,
            "n_clean_valid": int(g[clean_col].notna().sum()) if clean_col in g.columns else 0,
            "n_flagged_outliers": int(g[flag_col].sum()),
            "flagged_fraction": float(g[flag_col].mean()) if len(g) > 0 else np.nan,
        })
    return rows


# =====================================================
# MAIN
# =====================================================

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found:\n{INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    if DATE_COL not in df.columns:
        raise ValueError(f"Missing required date column: {DATE_COL}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).copy()

    missing_group_cols = [c for c in GROUP_COLS if c not in df.columns]
    if missing_group_cols:
        raise ValueError(f"Missing required group columns: {missing_group_cols}")

    df = ensure_s1_indicators(df)

    all_cleaned = []
    summary_rows = []

    for keys, g in df.groupby(GROUP_COLS, dropna=False):
        zone_id, part_id, source_name = keys
        g = g.sort_values(DATE_COL).copy()

        print("\nCleaning:")
        print(f"  zone_id={zone_id}, part_id={part_id}, source_name={source_name}, rows={len(g)}")

        g_clean = clean_group(g)

        safe_name = make_safe_name(source_name)
        out_csv = os.path.join(
            CLEAN_DIR,
            f"zone_{int(zone_id):02d}_part_{int(part_id)}_{safe_name}_sentinel1_cleaned.csv"
        )
        g_clean.to_csv(out_csv, index=False)
        print(f"  Saved cleaned CSV: {out_csv}")

        all_cleaned.append(g_clean)
        summary_rows.extend(summarize_cleaning(g_clean, zone_id, part_id, source_name))

        # ----------------- yearwise individual plots -----------------
        g_clean["Year"] = g_clean[DATE_COL].dt.year

        for year, d in g_clean.groupby("Year"):
            d = d.sort_values(DATE_COL).copy()
            base_title = f"Sentinel-1 Time Series: zone {zone_id}, {source_name}, part {part_id} ({int(year)})"

            for var_name, cfg in PLOT_CONFIGS.items():
                out_png = os.path.join(
                    PLOT_DIR,
                    f"zone_{int(zone_id):02d}_part_{int(part_id)}_{safe_name}_{int(year)}_{cfg['filename']}_raw_clean.png"
                )
                saved = save_individual_plot(d, cfg, base_title, out_png)
                if saved:
                    print(f"  Saved plot: {out_png}")

    if all_cleaned:
        combined = pd.concat(all_cleaned, ignore_index=True).sort_values(["zone_id", "part_id", DATE_COL])
        combined_csv = os.path.join(CLEAN_DIR, "all_zones_sentinel1_cleaned.csv")
        combined.to_csv(combined_csv, index=False)
        print(f"\nSaved combined cleaned CSV:\n{combined_csv}")

        summary = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(CLEAN_DIR, "sentinel1_cleaning_summary.csv")
        summary.to_csv(summary_csv, index=False)
        print(f"Saved cleaning summary CSV:\n{summary_csv}")
    else:
        print("No cleaned Sentinel-1 output created.")

    print("\nDone. Hare Krishna.")


if __name__ == "__main__":
    main()
