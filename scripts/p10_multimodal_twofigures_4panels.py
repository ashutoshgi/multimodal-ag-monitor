#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:11:10 2026

@author: ashutosh
"""

#Hare Krishna
#Hare Krishna

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create two yearly multimodal figures by zone, using 4 panels only.

Figure A: climate-context
1. NDVI
2. 7-day rainfall
3. Sentinel-1 selectable indicator
4. smoothed ETo + smoothed VPD

Figure B: response-oriented
1. NDVI
2. 7-day rainfall
3. Sentinel-1 selectable indicator
4. NDVI x ETo (optical dates only) + smoothed VPD

Inputs:
- multimodal merged CSV
- full daily weather CSV
- full cleaned Sentinel-1 CSV

Hare Krishna
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eo_config import OUTPUT_DIR, FIELD_NAME, EE_PROJECT, cleaned_shapefile_path, START_YEAR, END_YEAR, WEATHER_START_YEAR, WEATHER_END_YEAR, GRIDMET_BUFFER_M, CHIRPS_BUFFER_M, ERA5_BUFFER_M, LANDSAT_START_YEAR, LANDSAT_END_YEAR, LANDSAT_DOWNLOAD_IMAGES

# =====================================================
# USER SETTINGS
# =====================================================
#Hare Krishna
#Hare Krishna
#Hare Krishna
#Hare Krishna

BASE = OUTPUT_DIR

MULTIMODAL_CSV = os.path.join(BASE, "Multimodal", "all_zones_multimodal_merged.csv")
WEATHER_CSV = os.path.join(BASE, "Weather_TimeSeries", "all_zones_daily_weather_timeseries.csv")
S1_CSV = os.path.join(BASE, "Sentinel1_TimeSeries", "cleaned_per_zone_csv", "all_zones_sentinel1_cleaned.csv")

OUTPUT_DIR_A = os.path.join(BASE, "Multimodal", "plots_yearwise_figureA_climate_context_4panels")
OUTPUT_DIR_B = os.path.join(BASE, "Multimodal", "plots_yearwise_figureB_response_oriented_4panels")
os.makedirs(OUTPUT_DIR_A, exist_ok=True)
os.makedirs(OUTPUT_DIR_B, exist_ok=True)

# Weather variable selection
RAINFALL_COL = "gridmet_pr_mm"      # or "chirps_precip_mm"
ET_COL = "gridmet_eto_mm"
VPD_COL = "gridmet_vpd_kPa"

# Sentinel-1 variable selection
# Try:
# "VV_dB_clean"
# "VH_dB_clean"
# "VV_minus_VH_dB_clean"
# "VV_VH_ratio_linear_clean"
S1_COL = "VH_dB_clean"

# Plot / smoothing settings
FIGSIZE = (13, 11)
RAIN_ROLLING_DAYS = 7
ENV_ROLLING_DAYS = 7
S1_ROLLING_OBS = 5

# =====================================================
# HELPERS
# =====================================================
def make_safe_name(text):
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(text))


def require_file(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found:\n{path}")


def prep_df(path):
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"'date' column missing in file:\n{path}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    return df


def subset_zone_year(df, zone_id, part_id, year):
    out = df[(df["zone_id"] == zone_id) & (df["part_id"] == part_id)].copy()
    out = out[out["date"].dt.year == year].copy()
    out = out.sort_values("date")
    return out


def smooth_daily(series, window_days):
    s = pd.to_numeric(series, errors="coerce")
    return s.rolling(window=window_days, min_periods=1, center=True).mean()


def smooth_s1(series, window_obs):
    s = pd.to_numeric(series, errors="coerce")
    return s.rolling(window=window_obs, min_periods=1, center=True).median()


def get_s1_plot_column(d_s1, preferred_col):
    if preferred_col in d_s1.columns:
        return preferred_col
    raw_col = preferred_col.replace("_clean", "")
    if raw_col in d_s1.columns:
        return raw_col
    return None


def build_optical_eto_indicator(d_multi, d_weather, et_col):
    """
    Build NDVI x ETo for optical dates only.
    Uses nearest weather date merge within 2 days.
    Prefers Sentinel-2 NDVI if present, otherwise Landsat NDVI.
    """
    opt = d_multi.copy()

    opt["ndvi_for_indicator"] = np.nan
    if "s2_ndvi" in opt.columns:
        opt["ndvi_for_indicator"] = pd.to_numeric(opt["s2_ndvi"], errors="coerce")
    if "landsat_ndvi" in opt.columns:
        opt["ndvi_for_indicator"] = opt["ndvi_for_indicator"].fillna(
            pd.to_numeric(opt["landsat_ndvi"], errors="coerce")
        )

    opt = opt.dropna(subset=["date", "ndvi_for_indicator"]).copy()
    if opt.empty or et_col not in d_weather.columns:
        return pd.DataFrame(columns=["date", "ndvi_x_eto"])

    opt = opt.drop(columns=[et_col], errors="ignore")

    weather_small = d_weather[["date", et_col]].copy()
    weather_small[et_col] = pd.to_numeric(weather_small[et_col], errors="coerce")
    weather_small = weather_small.dropna(subset=["date"]).sort_values("date")

    opt = opt.sort_values("date")

    merged = pd.merge_asof(
        opt,
        weather_small,
        on="date",
        direction="nearest",
        tolerance=pd.Timedelta(days=2)
    )

    if et_col not in merged.columns:
        return pd.DataFrame(columns=["date", "ndvi_x_eto"])

    merged["ndvi_x_eto"] = (
        pd.to_numeric(merged["ndvi_for_indicator"], errors="coerce") *
        pd.to_numeric(merged[et_col], errors="coerce")
    )

    return merged[["date", "ndvi_x_eto"]].dropna()


def plot_common_panels(fig, axes, d_multi, d_weather, d_s1):
    # -------------------------------------------------
    # Panel 1: NDVI
    # -------------------------------------------------
    ndvi_plotted = False

    if "s2_ndvi" in d_multi.columns and d_multi["s2_ndvi"].notna().any():
        axes[0].plot(
            d_multi["date"], pd.to_numeric(d_multi["s2_ndvi"], errors="coerce"),
            marker="o", linewidth=2.0, label="Sentinel-2 NDVI"
        )
        ndvi_plotted = True

    if "landsat_ndvi" in d_multi.columns and d_multi["landsat_ndvi"].notna().any():
        axes[0].plot(
            d_multi["date"], pd.to_numeric(d_multi["landsat_ndvi"], errors="coerce"),
            marker="s", linewidth=1.8, label="Landsat NDVI"
        )
        ndvi_plotted = True

    axes[0].set_ylabel("NDVI")
    axes[0].set_ylim(-0.05, 1.0)
    axes[0].grid(True)
    if ndvi_plotted:
        axes[0].legend(loc="best")

    # -------------------------------------------------
    # Panel 2: 7-day rainfall accumulation
    # -------------------------------------------------
    rain_plotted = False
    if RAINFALL_COL in d_weather.columns and d_weather[RAINFALL_COL].notna().any():
        rain_daily = pd.to_numeric(d_weather[RAINFALL_COL], errors="coerce")
        rain_7d = rain_daily.rolling(RAIN_ROLLING_DAYS, min_periods=1).sum()

        axes[1].plot(
            d_weather["date"], rain_7d,
            linewidth=2.0, label=f"{RAINFALL_COL} {RAIN_ROLLING_DAYS}-day sum"
        )
        rain_plotted = True

    axes[1].set_ylabel("Rain (mm)")
    axes[1].grid(True)
    if rain_plotted:
        axes[1].legend(loc="best")

    # -------------------------------------------------
    # Panel 3: Sentinel-1 full observations
    # -------------------------------------------------
    s1_col_to_plot = get_s1_plot_column(d_s1, S1_COL)
    s1_plotted = False

    if s1_col_to_plot is not None and d_s1[s1_col_to_plot].notna().any():
        s1_raw = pd.to_numeric(d_s1[s1_col_to_plot], errors="coerce")
        s1_smoothed = smooth_s1(s1_raw, S1_ROLLING_OBS)

        axes[2].plot(
            d_s1["date"], s1_raw,
            linestyle="none", marker="o", markersize=5,
            alpha=0.7, label=f"{s1_col_to_plot} raw"
        )
        axes[2].plot(
            d_s1["date"], s1_smoothed,
            linewidth=2.0, label=f"{s1_col_to_plot} smoothed"
        )
        s1_plotted = True

    axes[2].set_ylabel("Sentinel-1")
    axes[2].grid(True)
    if s1_plotted:
        axes[2].legend(loc="best")


# =====================================================
# MAIN
# =====================================================
require_file(MULTIMODAL_CSV, "Multimodal CSV")
require_file(WEATHER_CSV, "Weather CSV")
require_file(S1_CSV, "Sentinel-1 cleaned CSV")

multi = prep_df(MULTIMODAL_CSV)
weather = prep_df(WEATHER_CSV)
s1 = prep_df(S1_CSV)

required_multi = ["zone_id", "part_id", "source_name", "date"]
missing_multi = [c for c in required_multi if c not in multi.columns]
if missing_multi:
    raise ValueError(f"Missing required columns in multimodal CSV: {missing_multi}")

required_weather = ["zone_id", "part_id", "source_name", "date"]
missing_weather = [c for c in required_weather if c not in weather.columns]
if missing_weather:
    raise ValueError(f"Missing required columns in weather CSV: {missing_weather}")

required_s1 = ["zone_id", "part_id", "source_name", "date"]
missing_s1 = [c for c in required_s1 if c not in s1.columns]
if missing_s1:
    raise ValueError(f"Missing required columns in Sentinel-1 CSV: {missing_s1}")

zone_keys = multi[["zone_id", "part_id", "source_name"]].drop_duplicates()

for _, zone_row in zone_keys.iterrows():
    zone_id = zone_row["zone_id"]
    part_id = zone_row["part_id"]
    source_name = str(zone_row["source_name"])
    safe_name = make_safe_name(source_name)

    zone_multi = multi[(multi["zone_id"] == zone_id) & (multi["part_id"] == part_id)].copy()
    years = sorted(zone_multi["date"].dt.year.dropna().unique())

    for year in years:
        d_multi = subset_zone_year(multi, zone_id, part_id, year)
        d_weather = subset_zone_year(weather, zone_id, part_id, year)
        d_s1 = subset_zone_year(s1, zone_id, part_id, year)

        if d_multi.empty and d_weather.empty and d_s1.empty:
            continue

        # =================================================
        # FIGURE A: climate-context
        # =================================================
        figA, axesA = plt.subplots(
            4, 1,
            figsize=FIGSIZE,
            sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.2, 1.4, 1.6]}
        )

        plot_common_panels(figA, axesA, d_multi, d_weather, d_s1)

        ax4A_right = axesA[3].twinx()

        if ET_COL in d_weather.columns and d_weather[ET_COL].notna().any():
            et_smooth = smooth_daily(d_weather[ET_COL], ENV_ROLLING_DAYS)
            axesA[3].plot(
                d_weather["date"], et_smooth,
                linewidth=2.0, label=f"{ET_COL} {ENV_ROLLING_DAYS}-day mean"
            )

        if VPD_COL in d_weather.columns and d_weather[VPD_COL].notna().any():
            vpd_smooth = smooth_daily(d_weather[VPD_COL], ENV_ROLLING_DAYS)
            ax4A_right.plot(
                d_weather["date"], vpd_smooth,
                linewidth=1.8, label=f"{VPD_COL} {ENV_ROLLING_DAYS}-day mean"
            )

        axesA[3].set_ylabel("ETo (mm)")
        ax4A_right.set_ylabel("VPD")
        axesA[3].grid(True)

        lines1, labels1 = axesA[3].get_legend_handles_labels()
        lines2, labels2 = ax4A_right.get_legend_handles_labels()
        if lines1 or lines2:
            axesA[3].legend(lines1 + lines2, labels1 + labels2, loc="best")

        axesA[3].set_xlabel("Date")

        figA.suptitle(
            f"Figure A: Climate-Context Multimodal Comparison\nZone {zone_id}, {source_name}, part {part_id} ({year})",
            fontsize=14
        )
        plt.tight_layout()

        out_png_A = os.path.join(
            OUTPUT_DIR_A,
            f"zone_{int(zone_id):02d}_part_{int(part_id)}_{safe_name}_{int(year)}_figureA_climate_context_4panels.png"
        )
        plt.savefig(out_png_A, dpi=200, bbox_inches="tight")
        plt.close(figA)
        print(f"Saved: {out_png_A}")

        # =================================================
        # FIGURE B: response-oriented
        # =================================================
        figB, axesB = plt.subplots(
            4, 1,
            figsize=FIGSIZE,
            sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.2, 1.4, 1.6]}
        )

        plot_common_panels(figB, axesB, d_multi, d_weather, d_s1)

        ax4B_right = axesB[3].twinx()

        ndvi_eto_df = build_optical_eto_indicator(d_multi, d_weather, ET_COL)
        if not ndvi_eto_df.empty:
            axesB[3].plot(
                ndvi_eto_df["date"], ndvi_eto_df["ndvi_x_eto"],
                marker="o", linewidth=1.8, label="NDVI x ETo"
            )

        if VPD_COL in d_weather.columns and d_weather[VPD_COL].notna().any():
            vpd_smooth = smooth_daily(d_weather[VPD_COL], ENV_ROLLING_DAYS)
            ax4B_right.plot(
                d_weather["date"], vpd_smooth,
                linewidth=1.8, label=f"{VPD_COL} {ENV_ROLLING_DAYS}-day mean"
            )

        axesB[3].set_ylabel("NDVI x ETo")
        ax4B_right.set_ylabel("VPD")
        axesB[3].grid(True)

        lines1, labels1 = axesB[3].get_legend_handles_labels()
        lines2, labels2 = ax4B_right.get_legend_handles_labels()
        if lines1 or lines2:
            axesB[3].legend(lines1 + lines2, labels1 + labels2, loc="best")

        axesB[3].set_xlabel("Date")

        figB.suptitle(
            f"Figure B: Response-Oriented Multimodal Comparison\nZone {zone_id}, {source_name}, part {part_id} ({year})",
            fontsize=14
        )
        plt.tight_layout()

        out_png_B = os.path.join(
            OUTPUT_DIR_B,
            f"zone_{int(zone_id):02d}_part_{int(part_id)}_{safe_name}_{int(year)}_figureB_response_oriented_4panels.png"
        )
        plt.savefig(out_png_B, dpi=200, bbox_inches="tight")
        plt.close(figB)
        print(f"Saved: {out_png_B}")

print("\nDone. Hare Krishna.")
