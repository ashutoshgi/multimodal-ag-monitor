#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot yearly Landsat NDVI time series for each polygon from cleaned CSV files.
No crop-window shading.

Hare Krishna
"""

#Hare Krishna

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

#Hare Krishna
#Hare Krishna
#Hare Krishna
#Hare Krishna

clean_folder = os.path.join(OUTPUT_DIR, "Landsat", "per_polygon_time_series", "cleaned_per_polygon_csv")


date_col = "date"
raw_col = "NDVI_raw_first_per_date"
orig_col = "NDVI_original_obs"
clean_col = "NDVI_clean"
qc_col = "QC_flag"

plot_folder = os.path.join(clean_folder, "plots_yearwise")
os.makedirs(plot_folder, exist_ok=True)


def make_safe_name(text):
    return ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in str(text))


clean_files = sorted(glob.glob(os.path.join(clean_folder, "zone_*_landsat_cleaned_C_QC.csv")))
if not clean_files:
    raise FileNotFoundError(f"No cleaned CSV files found in:\n{clean_folder}")

for clean_file in clean_files:
    df = pd.read_csv(clean_file)
    if date_col not in df.columns or clean_col not in df.columns:
        print(f"Skipped (missing required columns): {clean_file}")
        continue

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.floor("D")
    df = df.dropna(subset=[date_col, clean_col]).copy()
    if df.empty:
        print(f"Skipped empty file: {clean_file}")
        continue

    zone_id = df["zone_id"].iloc[0] if "zone_id" in df.columns else "NA"
    source_name = df["source_name"].iloc[0] if "source_name" in df.columns else "unknown"
    part_id = df["part_id"].iloc[0] if "part_id" in df.columns else "NA"
    safe_name = make_safe_name(source_name)

    df["Year"] = df[date_col].dt.year

    for year, d in df.groupby("Year"):
        d = d.sort_values(date_col)
        plt.figure(figsize=(11, 5))

        if raw_col in d.columns and d[raw_col].notna().any():
            plt.plot(d[date_col], d[raw_col], marker="o", linewidth=1, label="Raw first per date")

        if orig_col in d.columns and d[orig_col].notna().any():
            plt.plot(d[date_col], d[orig_col], marker="o", linewidth=1.5, label="Original kept obs")

        plt.plot(d[date_col], d[clean_col], marker="o", linewidth=2, label="Cleaned NDVI")

        if qc_col in d.columns:
            corrected = d[d[qc_col] == 1]
            if len(corrected) > 0:
                plt.scatter(corrected[date_col], corrected[clean_col], marker="x", s=80,
                            label="Corrected (QC flag=1)")

        plt.title(f"Landsat NDVI Time Series: zone {zone_id}, {source_name}, part {part_id} ({year})")
        plt.xlabel("Date")
        plt.ylabel("NDVI")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        outpng = os.path.join(
            plot_folder,
            f"zone_{int(zone_id):02d}_{safe_name}_part_{int(part_id)}_{int(year)}_LANDSAT_NDVI_raw_vs_clean_QC.png"
        )
        plt.savefig(outpng, dpi=200)
        plt.close()
        print(f"Saved: {outpng}")

print("\nDone. Year-wise Landsat NDVI plots saved in:")
print(plot_folder)
