#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared configuration loader for the multimodal EO time-series workflow."""

import os
from pathlib import Path
from datetime import datetime

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML is required. Install with: pip install pyyaml"
    ) from exc


def _find_config_path() -> Path:
    env_path = os.environ.get("EO_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()

    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "config" / "config.yaml",
        here.parents[1] / "config" / "config.example.yaml",
        Path.cwd() / "config" / "config.yaml",
        Path.cwd() / "config.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        "Could not find a config file. Copy config/config.example.yaml to "
        "config/config.yaml and edit it, or set EO_CONFIG=/path/to/config.yaml."
    )


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _cfg(path_keys, default=None):
    cur = CONFIG
    for key in path_keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _current_year_if_none(value):
    return datetime.now().year if value is None else int(value)


CONFIG_PATH = _find_config_path()
CONFIG = _load_yaml(CONFIG_PATH)

FIELD_NAME = str(_cfg(["project", "field_name"], "Example_Field"))
FIELD_SHP = str(Path(_cfg(["project", "field_shapefile"], "field_shapefile.shp")).expanduser())
OUTPUT_DIR = str(Path(_cfg(["project", "output_dir"], "Satellite_Data")).expanduser())
FIELD_ROOT = str(Path(FIELD_SHP).expanduser().resolve().parent)

# Optional crop mask. When enabled, p01 intersects the input field boundary
# with this mask and uses only the intersected/crop area for downloads and
# time-series extraction. Leave blank/null/None to process the full field.
APPLY_CROP_MASK = bool(_cfg(["mask", "apply_crop_mask"], False))
CROP_MASK_SHP = _cfg(["mask", "crop_mask_shapefile"], None)

if CROP_MASK_SHP in [None, "", "null", "None"]:
    CROP_MASK_SHP = None
else:
    CROP_MASK_SHP = str(Path(CROP_MASK_SHP).expanduser())

EE_PROJECT = _cfg(["gee", "project"], None)
if EE_PROJECT in ("", "null", "None"):
    EE_PROJECT = None

START_YEAR = int(_cfg(["season", "start_year"], 2020))
END_YEAR = _current_year_if_none(_cfg(["season", "end_year"], None))
SEASON_START_MONTH = int(_cfg(["season", "start_month"], 3))
SEASON_START_DAY = int(_cfg(["season", "start_day"], 1))
SEASON_END_MONTH = int(_cfg(["season", "end_month"], 10))
SEASON_END_DAY = int(_cfg(["season", "end_day"], 30))

S2_STEP_DAYS = int(_cfg(["sentinel2", "step_days"], 5))
S2_MAIN_CLOUD_FILTER = int(_cfg(["sentinel2", "main_cloud_filter"], 40))
S2_FALLBACK_CLOUD_FILTER = int(_cfg(["sentinel2", "fallback_cloud_filter"], 95))
RUN_SENTINEL2_SELECTION_AND_DOWNLOAD = bool(_cfg(["sentinel2", "run_selection_and_download"], True))
RUN_S2_POLYGON_TIMESERIES = bool(_cfg(["sentinel2", "run_polygon_timeseries"], True))
INNER_BUFFER_METERS = _cfg(["sentinel2", "inner_buffer_meters"], None)

S1_STEP_DAYS = int(_cfg(["sentinel1", "step_days"], 5))
S1_ORBIT = _cfg(["sentinel1", "orbit"], None)
if S1_ORBIT in ("", "null", "None"):
    S1_ORBIT = None
RUN_SENTINEL1_DOWNLOAD = bool(_cfg(["sentinel1", "run_download"], True))

WEATHER_START_YEAR = int(_cfg(["weather", "start_year"], 2000))
WEATHER_END_YEAR = _current_year_if_none(_cfg(["weather", "end_year"], None))
GRIDMET_BUFFER_M = int(_cfg(["weather", "gridmet_buffer_m"], 5000))
CHIRPS_BUFFER_M = int(_cfg(["weather", "chirps_buffer_m"], 6000))
ERA5_BUFFER_M = int(_cfg(["weather", "era5_buffer_m"], 12000))

LANDSAT_START_YEAR = int(_cfg(["landsat", "start_year"], 1984))
LANDSAT_END_YEAR = _current_year_if_none(_cfg(["landsat", "end_year"], None))
LANDSAT_DOWNLOAD_IMAGES = bool(_cfg(["landsat", "download_images"], False))

DPI = int(_cfg(["plotting", "dpi"], 300))


def cleaned_shapefile_path() -> str:
    return os.path.join(OUTPUT_DIR, f"{FIELD_NAME}_cleaned_polygons.shp")


def print_config_summary():
    print("=" * 70)
    print("Configuration")
    print(f"Config file: {CONFIG_PATH}")
    print(f"Field name : {FIELD_NAME}")
    print(f"Field shp  : {FIELD_SHP}")
    print(f"Crop mask  : {CROP_MASK_SHP if APPLY_CROP_MASK else 'not used'}")
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"EE project : {EE_PROJECT}")
    print("=" * 70)
