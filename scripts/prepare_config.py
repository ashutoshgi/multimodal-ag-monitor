#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a config/config.yaml file for the multimodal EO time-series workflow.

Example:
python scripts/prepare_config.py \
  --field-name Example \
  --field-shapefile "/path/to/field_shapefile.shp" \
  --output-dir "/path/to/Satellite_Data" \
  --ee-project "your-earth-engine-project-id" \
  --write config/config.yaml

The script does not download data. It only writes a YAML config file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc


def infer_field_name(shapefile: Path) -> str:
    # Prefer the parent folder name unless it is a generic Shapefile folder.
    parent = shapefile.parent
    if parent.name.lower() in {"shapefile", "shapefiles"} and parent.parent.name:
        return parent.parent.name
    return parent.name or shapefile.stem


def infer_output_dir(shapefile: Path) -> Path:
    # If path ends with Satellite/Shapefile/field_shapefile.shp, use Satellite/Satellite_Data.
    parent = shapefile.parent
    if parent.name.lower() in {"shapefile", "shapefiles"}:
        return parent.parent / "Satellite_Data"
    return parent / "Satellite_Data"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    v = value.strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def build_config(args: argparse.Namespace) -> dict:
    shp = Path(args.field_shapefile).expanduser()
    if not shp.suffix.lower() == ".shp":
        raise ValueError("--field-shapefile should point to a .shp file")

    field_name = args.field_name or infer_field_name(shp)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else infer_output_dir(shp)

    crop_mask = None
    if args.crop_mask_shapefile:
        crop_mask = str(Path(args.crop_mask_shapefile).expanduser())

    return {
        "project": {
            "field_name": field_name,
            "field_shapefile": str(shp),
            "output_dir": str(output_dir),
        },
        "mask": {
            "apply_crop_mask": bool(crop_mask),
            "crop_mask_shapefile": crop_mask,
        },
        "gee": {
            "project": None if args.ee_project in {"", "null", "None", None} else args.ee_project,
        },
        "season": {
            "start_year": args.start_year,
            "end_year": None if args.end_year in {0, None} else args.end_year,
            "start_month": args.season_start_month,
            "start_day": args.season_start_day,
            "end_month": args.season_end_month,
            "end_day": args.season_end_day,
        },
        "sentinel2": {
            "step_days": args.s2_step_days,
            "main_cloud_filter": args.s2_main_cloud_filter,
            "fallback_cloud_filter": args.s2_fallback_cloud_filter,
            "run_selection_and_download": args.run_sentinel2,
            "run_polygon_timeseries": args.run_s2_timeseries,
            "inner_buffer_meters": args.inner_buffer_meters,
        },
        "sentinel1": {
            "step_days": args.s1_step_days,
            "orbit": args.s1_orbit,
            "run_download": args.run_sentinel1,
        },
        "weather": {
            "start_year": args.weather_start_year,
            "end_year": None if args.weather_end_year in {0, None} else args.weather_end_year,
            "gridmet_buffer_m": args.gridmet_buffer_m,
            "chirps_buffer_m": args.chirps_buffer_m,
            "era5_buffer_m": args.era5_buffer_m,
        },
        "landsat": {
            "start_year": args.landsat_start_year,
            "end_year": None if args.landsat_end_year in {0, None} else args.landsat_end_year,
            "download_images": args.download_landsat_images,
        },
        "plotting": {
            "dpi": args.dpi,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create config.yaml for the multimodal EO time-series workflow.")
    parser.add_argument("--field-shapefile", required=True, help="Full path to field_shapefile.shp")
    parser.add_argument("--field-name", default=None, help="Field name. If omitted, inferred from the shapefile folder.")
    parser.add_argument("--output-dir", default=None, help="Output Satellite_Data directory. If omitted, inferred from shapefile path.")
    parser.add_argument("--crop-mask-shapefile", default=None, help="Optional crop mask shapefile. If provided, crop masking is enabled.")
    parser.add_argument("--ee-project", default="your-earth-engine-project-id", help="Google Earth Engine project id, or null.")
    parser.add_argument("--write", default="config/config.yaml", help="Output YAML config path.")

    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=0, help="0 = current year")
    parser.add_argument("--season-start-month", type=int, default=3)
    parser.add_argument("--season-start-day", type=int, default=1)
    parser.add_argument("--season-end-month", type=int, default=10)
    parser.add_argument("--season-end-day", type=int, default=30)

    parser.add_argument("--s2-step-days", type=int, default=5)
    parser.add_argument("--s2-main-cloud-filter", type=int, default=40)
    parser.add_argument("--s2-fallback-cloud-filter", type=int, default=95)
    parser.add_argument("--run-sentinel2", type=parse_bool, default=True)
    parser.add_argument("--run-s2-timeseries", type=parse_bool, default=True)
    parser.add_argument("--inner-buffer-meters", type=float, default=None)

    parser.add_argument("--s1-step-days", type=int, default=5)
    parser.add_argument("--s1-orbit", default=None, choices=[None, "ASCENDING", "DESCENDING"], help="Optional Sentinel-1 orbit filter")
    parser.add_argument("--run-sentinel1", type=parse_bool, default=True)

    parser.add_argument("--weather-start-year", type=int, default=2000)
    parser.add_argument("--weather-end-year", type=int, default=0, help="0 = current year")
    parser.add_argument("--gridmet-buffer-m", type=int, default=5000)
    parser.add_argument("--chirps-buffer-m", type=int, default=6000)
    parser.add_argument("--era5-buffer-m", type=int, default=12000)

    parser.add_argument("--landsat-start-year", type=int, default=1984)
    parser.add_argument("--landsat-end-year", type=int, default=0, help="0 = current year")
    parser.add_argument("--download-landsat-images", type=parse_bool, default=False)
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()
    config = build_config(args)

    out_path = Path(args.write).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    print(f"Saved config: {out_path.resolve()}")
    print("Field:", config["project"]["field_name"])
    print("Shapefile:", config["project"]["field_shapefile"])
    print("Output dir:", config["project"]["output_dir"])
    print("Crop mask:", config["mask"]["crop_mask_shapefile"] or "not used")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
