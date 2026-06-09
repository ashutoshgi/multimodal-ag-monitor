#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 2026

Plot year-wise Sentinel-2 index time series for each polygon from cleaned CSV files,
with zone-wise mean curve overlaid on the same plot.

Expected input:
    cleaned CSV files produced by gaurnitai_remove_outliers_all_indices_by_ndvi.py
    File pattern:
        zone_*_all_indices_cleaned_by_NDVI.csv

Outputs:
    - one yearly plot per polygon per index
    - NDVI: original kept obs, cleaned NDVI, QC-corrected points, zone mean overlay
    - SAVI/NDWI/NDRE: kept obs and zone mean overlay

"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- SETTINGS ----------------


clean_folder = os.path.join(OUTPUT_DIR, "Sentinel2", "per_polygon_time_series", "cleaned_per_polygon_csv")

# Input file pattern from NDVI-based all-index cleaning step
clean_files_pattern = "zone_*_all_indices_cleaned_by_NDVI.csv"

date_col = "date"
qc_col = "QC_flag"

# What to plot for each index
# NDVI has a dedicated cleaned column; others only have retained observations
index_configs = {
    "NDVI": {
        "raw_col": "NDVI_raw_first_per_date",
        "orig_col": "NDVI_original_obs",
        "plot_col": "NDVI_clean",
        "zone_mean_col": "NDVI_clean",
        "ylabel": "NDVI",
        "title_name": "NDVI",
        "show_raw": True,
        "show_orig": True,
        "show_qc": True,
    },
    "SAVI": {
        "raw_col": "SAVI_raw_first_per_date",
        "orig_col": "SAVI_original_obs",
        "plot_col": "SAVI_original_obs",
        "zone_mean_col": "SAVI_original_obs",
        "ylabel": "SAVI",
        "title_name": "SAVI",
        "show_raw": False,
        "show_orig": True,
        "show_qc": False,
    },
    "NDWI": {
        "raw_col": "NDWI_raw_first_per_date",
        "orig_col": "NDWI_original_obs",
        "plot_col": "NDWI_original_obs",
        "zone_mean_col": "NDWI_original_obs",
        "ylabel": "NDWI",
        "title_name": "NDWI",
        "show_raw": False,
        "show_orig": True,
        "show_qc": False,
    },
    "NDRE": {
        "raw_col": "NDRE_raw_first_per_date",
        "orig_col": "NDRE_original_obs",
        "plot_col": "NDRE_original_obs",
        "zone_mean_col": "NDRE_original_obs",
        "ylabel": "NDRE",
        "title_name": "NDRE",
        "show_raw": False,
        "show_orig": True,
        "show_qc": False,
    },
}

plot_root = os.path.join(clean_folder, "plots_yearwise_with_zone_mean")
os.makedirs(plot_root, exist_ok=True)

# ------------------------------------------
def make_safe_name(text):
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(text))


def read_all_clean_files(clean_files):
    dfs = []
    for f in clean_files:
        try:
            df = pd.read_csv(f)
            df["_source_csv"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"Skipped unreadable file: {f}\n  Reason: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


clean_files = sorted(glob.glob(os.path.join(clean_folder, clean_files_pattern)))
if not clean_files:
    raise FileNotFoundError(
        f"No cleaned CSV files found in:\n{clean_folder}\nPattern: {clean_files_pattern}"
    )

# Read all files once so zone-wise means can be computed globally across polygons
all_df = read_all_clean_files(clean_files)
if all_df.empty:
    raise RuntimeError("No valid cleaned CSV files could be read.")

if date_col not in all_df.columns or "zone_id" not in all_df.columns:
    raise ValueError(f"Required columns missing from combined data. Found columns: {list(all_df.columns)}")

all_df[date_col] = pd.to_datetime(all_df[date_col], errors="coerce").dt.floor("D")
all_df = all_df.dropna(subset=[date_col]).copy()
all_df["Year"] = all_df[date_col].dt.year

# Precompute zone-wise mean per date for each index
zone_mean_tables = {}
for idx_name, cfg in index_configs.items():
    zcol = cfg["zone_mean_col"]
    if zcol not in all_df.columns:
        print(f"Warning: zone-mean source column missing for {idx_name}: {zcol}")
        continue

    temp = all_df[["zone_id", date_col, "Year", zcol]].copy()
    temp[zcol] = pd.to_numeric(temp[zcol], errors="coerce")
    temp = temp.dropna(subset=[zcol])

    zone_mean = (
        temp.groupby(["zone_id", date_col, "Year"], as_index=False)[zcol]
        .mean()
        .rename(columns={zcol: "zone_mean_value"})
    )
    zone_mean_tables[idx_name] = zone_mean

# Plot each polygon file for each year and each index
for clean_file in clean_files:
    df = pd.read_csv(clean_file)

    if date_col not in df.columns:
        print(f"Skipped (missing date column): {clean_file}")
        continue

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.floor("D")
    df = df.dropna(subset=[date_col]).copy()

    if df.empty:
        print(f"Skipped empty file: {clean_file}")
        continue

    zone_id = df["zone_id"].iloc[0] if "zone_id" in df.columns else "NA"
    source_name = df["source_name"].iloc[0] if "source_name" in df.columns else "unknown"
    part_id = df["part_id"].iloc[0] if "part_id" in df.columns else "NA"

    safe_name = make_safe_name(source_name)
    df["Year"] = df[date_col].dt.year

    for idx_name, cfg in index_configs.items():
        plot_col = cfg["plot_col"]
        raw_col = cfg["raw_col"]
        orig_col = cfg["orig_col"]

        if plot_col not in df.columns and orig_col not in df.columns:
            print(f"Skipped {idx_name} for {os.path.basename(clean_file)} (missing plot columns)")
            continue

        idx_plot_folder = os.path.join(plot_root, idx_name)
        os.makedirs(idx_plot_folder, exist_ok=True)

        for year, d in df.groupby("Year"):
            d = d.sort_values(date_col).copy()

            # convert numeric safely
            for c in [raw_col, orig_col, plot_col, qc_col]:
                if c in d.columns:
                    d[c] = pd.to_numeric(d[c], errors="coerce")

            # choose main polygon series
            main_series_col = plot_col if plot_col in d.columns else orig_col
            d = d.dropna(subset=[main_series_col])
            if d.empty:
                continue

            plt.figure(figsize=(11.5, 5.5))

            if cfg["show_raw"] and raw_col in d.columns and d[raw_col].notna().any():
                plt.plot(d[date_col], d[raw_col], marker="o", linewidth=1, label=f"{idx_name} raw first per date")

            if cfg["show_orig"] and orig_col in d.columns and d[orig_col].notna().any():
                label = f"{idx_name} original kept obs" if idx_name == "NDVI" else f"{idx_name} polygon"
                plt.plot(d[date_col], d[orig_col], marker="o", linewidth=1.4, label=label)

            if plot_col in d.columns and plot_col != orig_col and d[plot_col].notna().any():
                plt.plot(d[date_col], d[plot_col], marker="o", linewidth=2.2, label=f"Cleaned {idx_name}")

            # QC-corrected points (NDVI only)
            if cfg["show_qc"] and qc_col in d.columns and plot_col in d.columns:
                corrected = d[(d[qc_col] == 1) & d[plot_col].notna()]
                if len(corrected) > 0:
                    plt.scatter(
                        corrected[date_col], corrected[plot_col],
                        marker="x", s=80, label="Corrected (QC flag=1)"
                    )

            # Zone-wise mean overlay
            zmean = zone_mean_tables.get(idx_name)
            if zmean is not None:
                z = zmean[(zmean["zone_id"] == zone_id) & (zmean["Year"] == year)].copy()
                z = z.sort_values(date_col)
                if not z.empty:
                    plt.plot(
                        z[date_col], z["zone_mean_value"],
                        linestyle="--", linewidth=2.4,
                        label=f"Zone {int(zone_id):02d} mean {idx_name}"
                    )

            plt.title(f"{cfg['title_name']} Time Series: zone {zone_id}, {source_name}, part {part_id} ({year})")
            plt.xlabel("Date")
            plt.ylabel(cfg["ylabel"])
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            outpng = os.path.join(
                idx_plot_folder,
                f"zone_{int(zone_id):02d}_{safe_name}_part_{int(part_id)}_{int(year)}_{idx_name}_with_zone_mean.png"
            )
            plt.savefig(outpng, dpi=200)
            plt.close()
            print(f"Saved: {outpng}")

print("\nDone. Year-wise index plots with zone-wise mean overlays saved in:")
print(plot_root)
