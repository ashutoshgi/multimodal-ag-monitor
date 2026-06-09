#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:57:34 2026

@author: ashutosh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create yearly pixelwise NDVI spaghetti plots for each cleaned zone polygon,
using only dates retained after NDVI outlier cleaning.

Output:
    one spaghetti plot per zone per year

"""

import os
import re
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt

from eo_config import OUTPUT_DIR, FIELD_NAME, EE_PROJECT, cleaned_shapefile_path, START_YEAR, END_YEAR, WEATHER_START_YEAR, WEATHER_END_YEAR, GRIDMET_BUFFER_M, CHIRPS_BUFFER_M, ERA5_BUFFER_M, LANDSAT_START_YEAR, LANDSAT_END_YEAR, LANDSAT_DOWNLOAD_IMAGES

# =====================================================
# USER SETTINGS
# =====================================================

#Hare Krishna

#Hare Krishna

#Hare Krishna

#Hare Krishna

BASE_DIR = OUTPUT_DIR

S2_IMAGE_DIR = os.path.join(BASE_DIR, "Sentinel2", "images")
CLEANED_SHP = cleaned_shapefile_path()

CLEANED_CSV = os.path.join(
    BASE_DIR,
    "Sentinel2",
    "per_polygon_time_series",
    "cleaned_per_polygon_csv",
    "all_polygons_NDVI_cleaned_C_QC.csv"
)


OUTPUT_DIR = os.path.join(BASE_DIR, "Sentinel2", "zonewise_pixelwise_ndvi_spaghetti")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Date logic:
# "intersection" = dates retained in all polygons
# "union"        = dates retained in at least one polygon
KEEP_DATES_MODE = "union"

# Downloaded TIFF band order:
# ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
RED_BAND_INDEX_1BASED = 4
NIR_BAND_INDEX_1BASED = 8

# Plot controls
MAX_PIXELS_TO_PLOT = 500
RANDOM_SEED = 42
MIN_VALID_DATES_PER_PIXEL = 3
FIGSIZE = (11, 6)
SAVE_MEAN_LINE = True

# =====================================================
# HELPERS
# =====================================================
def compute_ndvi(red, nir):
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    np.seterr(divide="ignore", invalid="ignore")
    denom = nir + red
    ndvi = np.where(denom != 0, (nir - red) / denom, np.nan)
    ndvi[~np.isfinite(ndvi)] = np.nan
    return ndvi.astype(np.float32)


def extract_date_from_filename(path):
    base = os.path.basename(path)
    m = re.search(r"Sentinel2_(\d{4}-\d{2}-\d{2})\.tif$", base)
    if m:
        return pd.to_datetime(m.group(1), errors="coerce").strftime("%Y-%m-%d")
    return None


def make_safe_name(text):
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(text))


def detect_source_name_column(gdf):
    for col in ["source_name", "source_nam", "source", "name", "Name", "NAME"]:
        if col in gdf.columns:
            return col
    return None


def get_kept_dates(cleaned_csv, mode="intersection"):
    df = pd.read_csv(cleaned_csv)

    required = ["zone_id", "part_id", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in cleaned CSV: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    group_keys = df[["zone_id", "part_id"]].drop_duplicates()

    if mode.lower() == "union":
        kept_dates = set(df["date_str"].unique())

    elif mode.lower() == "intersection":
        date_sets = []
        for _, row in group_keys.iterrows():
            sub = df[
                (df["zone_id"] == row["zone_id"]) &
                (df["part_id"] == row["part_id"])
            ]
            date_sets.append(set(sub["date_str"].unique()))
        kept_dates = set.intersection(*date_sets) if date_sets else set()

    else:
        raise ValueError("KEEP_DATES_MODE must be either 'intersection' or 'union'.")

    kept_dates = sorted(kept_dates)
    print(f"Kept dates mode: {mode}")
    print(f"Number of kept dates: {len(kept_dates)}")
    return kept_dates


def get_selected_tifs(image_dir, kept_dates_set):
    tif_files = sorted(glob.glob(os.path.join(image_dir, "Sentinel2_*.tif")))
    selected = []
    for tif_path in tif_files:
        d = extract_date_from_filename(tif_path)
        if d is not None and d in kept_dates_set:
            selected.append((pd.to_datetime(d), tif_path))
    return sorted(selected, key=lambda x: x[0])


def sample_pixel_indices(valid_mask, max_pixels, random_seed=42):
    idx = np.argwhere(valid_mask)
    if len(idx) == 0:
        return np.empty((0, 2), dtype=int)

    rng = np.random.default_rng(random_seed)
    if len(idx) <= max_pixels:
        return idx
    chosen = rng.choice(len(idx), size=max_pixels, replace=False)
    return idx[chosen]


# =====================================================
# MAIN
# =====================================================
if not os.path.exists(CLEANED_SHP):
    raise FileNotFoundError(f"Cleaned shapefile not found:\n{CLEANED_SHP}")

if not os.path.exists(CLEANED_CSV):
    raise FileNotFoundError(f"Cleaned NDVI CSV not found:\n{CLEANED_CSV}")

if not os.path.exists(S2_IMAGE_DIR):
    raise FileNotFoundError(f"Sentinel-2 image folder not found:\n{S2_IMAGE_DIR}")

zones_gdf = gpd.read_file(CLEANED_SHP)
if zones_gdf.empty:
    raise ValueError("Cleaned polygon shapefile is empty.")
if zones_gdf.crs is None:
    raise ValueError("Cleaned polygon shapefile has no CRS.")

required_zone_cols = ["zone_id", "part_id", "geometry"]
missing_zone_cols = [c for c in required_zone_cols if c not in zones_gdf.columns]
if missing_zone_cols:
    raise ValueError(f"Missing required columns in cleaned shapefile: {missing_zone_cols}")

source_name_col = detect_source_name_column(zones_gdf)
if source_name_col is None:
    zones_gdf["source_label"] = zones_gdf.apply(
        lambda r: f"zone_{int(r['zone_id']):02d}_part_{int(r['part_id'])}",
        axis=1
    )
else:
    zones_gdf["source_label"] = zones_gdf[source_name_col].astype(str)

kept_dates = get_kept_dates(CLEANED_CSV, mode=KEEP_DATES_MODE)
kept_dates_set = set(kept_dates)

selected_tifs = get_selected_tifs(S2_IMAGE_DIR, kept_dates_set)
print(f"Selected TIFFs after date filtering: {len(selected_tifs)}")

if len(selected_tifs) == 0:
    raise ValueError("No Sentinel-2 TIFFs matched the kept dates from cleaned CSV.")

# Group selected TIFFs by year
year_to_tifs = {}
for dt, tif_path in selected_tifs:
    year_to_tifs.setdefault(dt.year, []).append((dt, tif_path))

for _, row in zones_gdf.iterrows():
    zone_id = int(row["zone_id"])
    part_id = int(row["part_id"])
    source_name = str(row["source_label"])
    geom = row.geometry

    if geom is None or geom.is_empty:
        continue

    safe_name = make_safe_name(source_name)
    print(f"\nProcessing zone {zone_id}, part {part_id}, label={source_name}")

    for year, tif_list in sorted(year_to_tifs.items()):
        print(f"  Year {year}: {len(tif_list)} dates")

        date_list = []
        ndvi_stack = []
        ref_shape = None

        for dt, tif_path in tif_list:
            with rasterio.open(tif_path) as src:
                if zones_gdf.crs != src.crs:
                    geom_here = gpd.GeoSeries([geom], crs=zones_gdf.crs).to_crs(src.crs).iloc[0]
                else:
                    geom_here = geom

                try:
                    cropped_red, _ = mask(
                        src, [geom_here], crop=True, filled=False,
                        indexes=RED_BAND_INDEX_1BASED
                    )
                    cropped_nir, _ = mask(
                        src, [geom_here], crop=True, filled=False,
                        indexes=NIR_BAND_INDEX_1BASED
                    )

                    cropped_red = np.ma.filled(cropped_red.astype(np.float32), np.nan)
                    cropped_nir = np.ma.filled(cropped_nir.astype(np.float32), np.nan)
                    cropped_ndvi = compute_ndvi(cropped_red, cropped_nir)

                    if ref_shape is None:
                        ref_shape = cropped_ndvi.shape

                    if cropped_ndvi.shape != ref_shape:
                        print(f"    Skipping {dt.date()} due to shape mismatch: {cropped_ndvi.shape} vs {ref_shape}")
                        continue

                    if np.all(np.isnan(cropped_ndvi)):
                        print(f"    Skipping {dt.date()} because NDVI is all-NaN")
                        continue

                    ndvi_stack.append(cropped_ndvi)
                    date_list.append(dt)

                except Exception as e:
                    print(f"    Failed {dt.date()}: {e}")

        if len(ndvi_stack) == 0:
            print(f"    No valid NDVI data for zone {zone_id}, year {year}")
            continue

        ndvi_stack = np.stack(ndvi_stack, axis=0)   # (time, rows, cols)
        dates = pd.to_datetime(date_list)

        # Keep pixels having enough valid dates
        valid_counts = np.sum(np.isfinite(ndvi_stack), axis=0)
        valid_mask = valid_counts >= MIN_VALID_DATES_PER_PIXEL

        sampled_pixels = sample_pixel_indices(valid_mask, MAX_PIXELS_TO_PLOT, RANDOM_SEED)

        if len(sampled_pixels) == 0:
            print(f"    No valid pixels to plot for zone {zone_id}, year {year}")
            continue

        plt.figure(figsize=FIGSIZE)

        for r_idx, c_idx in sampled_pixels:
            ts = ndvi_stack[:, r_idx, c_idx]
            ok = np.isfinite(ts)
            if np.sum(ok) >= MIN_VALID_DATES_PER_PIXEL:
                plt.plot(dates[ok], ts[ok], linewidth=0.6, alpha=0.25)

        if SAVE_MEAN_LINE:
            mean_ts = np.nanmean(ndvi_stack, axis=(1, 2))
            okm = np.isfinite(mean_ts)
            if np.sum(okm) > 0:
                plt.plot(dates[okm], mean_ts[okm], linewidth=2.5, label="Mean NDVI")

        plt.title(f"Pixelwise NDVI Spaghetti Plot\nZone {zone_id}, {source_name}, part {part_id} ({year})")
        plt.xlabel("Date")
        plt.ylabel("NDVI")
        plt.ylim(-0.05, 1.0)
        plt.grid(True)
        if SAVE_MEAN_LINE:
            plt.legend()
        plt.tight_layout()

        out_png = os.path.join(
            OUTPUT_DIR,
            f"zone_{zone_id:02d}_part_{part_id}_{safe_name}_{year}_pixelwise_NDVI_spaghetti.png"
        )
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"    Saved: {out_png}")

print("\nDone. Hare Krishna.")
