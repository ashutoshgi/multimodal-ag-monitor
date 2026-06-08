#Hare Krishna

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create yearly multimodal comparison plots by zone using:
- NDVI from multimodal merged table
- full daily weather from weather CSV
- full Sentinel-1 observations from cleaned Sentinel-1 CSV

Panels:
1. NDVI (Sentinel-2 and Landsat)
2. 7-day rainfall accumulation
3. Sentinel-1 selectable indicator (raw points + smoothed line)
4. ET and VPD (7-day smoothed)

Hare Krishna
"""

import os
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

OUTPUT_DIR = os.path.join(BASE, "Multimodal", "plots_yearwise_full_context")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Weather variables
RAINFALL_COL = "gridmet_pr_mm"       # alternative: "chirps_precip_mm"
ET_COL = "gridmet_eto_mm"
VPD_COL = "gridmet_vpd_kPa"

# Sentinel-1 variable to experiment with:
# "VV_dB_clean"
# "VH_dB_clean"
# "VV_minus_VH_dB_clean"
# "VV_VH_ratio_linear_clean"
S1_COL = "VH_dB_clean"

# Plot settings
FIGSIZE = (13, 11)

# Smoothing settings
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


def smooth_time_series(series, window):
    s = pd.to_numeric(series, errors="coerce")
    return s.rolling(window=window, min_periods=1, center=True).mean()


def smooth_s1_series(series, window_obs):
    s = pd.to_numeric(series, errors="coerce")
    return s.rolling(window=window_obs, min_periods=1, center=True).median()


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

        fig, axes = plt.subplots(
            4, 1,
            figsize=FIGSIZE,
            sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.3, 1.5, 1.5]}
        )

        # -------------------------------------------------
        # Panel 1: NDVI
        # -------------------------------------------------
        ndvi_plotted = False

        if "s2_ndvi" in d_multi.columns and d_multi["s2_ndvi"].notna().any():
            axes[0].plot(
                d_multi["date"], d_multi["s2_ndvi"],
                marker="o", linewidth=2.0, label="Sentinel-2 NDVI"
            )
            ndvi_plotted = True

        if "landsat_ndvi" in d_multi.columns and d_multi["landsat_ndvi"].notna().any():
            axes[0].plot(
                d_multi["date"], d_multi["landsat_ndvi"],
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
        s1_plotted = False

        plot_s1_col = S1_COL
        if plot_s1_col not in d_s1.columns and plot_s1_col.replace("_clean", "") in d_s1.columns:
            plot_s1_col = plot_s1_col.replace("_clean", "")

        if plot_s1_col in d_s1.columns and d_s1[plot_s1_col].notna().any():
            s1_raw = pd.to_numeric(d_s1[plot_s1_col], errors="coerce")
            s1_smooth = smooth_s1_series(s1_raw, S1_ROLLING_OBS)

            # raw points
            axes[2].plot(
                d_s1["date"], s1_raw,
                linestyle="none", marker="o", markersize=5,
                alpha=0.7, label=f"{plot_s1_col} raw"
            )

            # smoothed line
            axes[2].plot(
                d_s1["date"], s1_smooth,
                linewidth=2.0, label=f"{plot_s1_col} smoothed"
            )

            s1_plotted = True

        axes[2].set_ylabel("Sentinel-1")
        axes[2].grid(True)
        if s1_plotted:
            axes[2].legend(loc="best")

        # -------------------------------------------------
        # Panel 4: ET and VPD full daily, smoothed
        # -------------------------------------------------
        env_plotted = False

        ax4b = axes[3].twinx()

        if ET_COL in d_weather.columns and d_weather[ET_COL].notna().any():
            et = smooth_time_series(d_weather[ET_COL], ENV_ROLLING_DAYS)
            axes[3].plot(
                d_weather["date"], et,
                linewidth=2.0, label=f"{ET_COL} {ENV_ROLLING_DAYS}-day mean"
            )
            env_plotted = True

        if VPD_COL in d_weather.columns and d_weather[VPD_COL].notna().any():
            vpd = smooth_time_series(d_weather[VPD_COL], ENV_ROLLING_DAYS)
            ax4b.plot(
                d_weather["date"], vpd,
                linewidth=1.8, label=f"{VPD_COL} {ENV_ROLLING_DAYS}-day mean"
            )
            env_plotted = True

        axes[3].set_ylabel("ET (mm)")
        ax4b.set_ylabel("VPD")

        lines1, labels1 = axes[3].get_legend_handles_labels()
        lines2, labels2 = ax4b.get_legend_handles_labels()
        if lines1 or lines2:
            axes[3].legend(lines1 + lines2, labels1 + labels2, loc="best")

        axes[3].grid(True)
        axes[3].set_xlabel("Date")

        fig.suptitle(
            f"Multimodal Yearly Comparison\nZone {zone_id}, {source_name}, part {part_id} ({year})",
            fontsize=14
        )
        plt.tight_layout()

        out_png = os.path.join(
            OUTPUT_DIR,
            f"zone_{int(zone_id):02d}_part_{int(part_id)}_{safe_name}_{int(year)}_multimodal_full_context.png"
        )
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {out_png}")

print("\nDone. Hare Krishna.")
