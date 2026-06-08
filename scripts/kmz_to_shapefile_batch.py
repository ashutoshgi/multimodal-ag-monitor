#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kmz_to_shapefile_batch.py

Convert a directory of KMZ/KML files into shapefiles usable by the
multimodal EO time-series repository.

Main outputs:
1. One shapefile per KMZ/KML:
   <output_root>/<safe_kmz_name>/field_shapefile.shp

2. One combined shapefile:
   <output_root>/all_kmz_polygons_combined.shp

Optional repository-ready folder structure:
   <output_root>/<safe_kmz_name>/Satellite/Shapefile/field_shapefile.shp
   <output_root>/<safe_kmz_name>/Satellite/Satellite_Data/

Hare Krishna
"""
#Hare Krishna
#Hare Krishna
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid


DEFAULT_INPUT_DIR = None
DEFAULT_OUTPUT_ROOT = None


def safe_name(text: str, max_len: int = 80) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", text)
    text = text.strip("_.-")
    if not text:
        text = "unnamed"
    return text[:max_len]


def strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def child_text(elem: ET.Element, child_name: str) -> str | None:
    for child in list(elem):
        if strip_namespace(child.tag) == child_name:
            return child.text
    return None


def iter_children_by_name(elem: ET.Element, name: str):
    for child in elem.iter():
        if strip_namespace(child.tag) == name:
            yield child


def parse_coord_text(coord_text: str):
    """Parse a KML coordinate string into [(lon, lat), ...]."""
    coords = []
    if coord_text is None:
        return coords
    for token in coord_text.replace("\n", " ").replace("\t", " ").split():
        parts = token.split(",")
        if len(parts) < 2:
            continue
        try:
            lon = float(parts[0])
            lat = float(parts[1])
            coords.append((lon, lat))
        except ValueError:
            continue
    return coords


def close_ring(coords):
    if len(coords) >= 3 and coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    return coords


def polygon_from_kml_polygon(poly_elem: ET.Element):
    outer = None
    holes = []

    for outer_elem in iter_children_by_name(poly_elem, "outerBoundaryIs"):
        coords_elem = next(iter_children_by_name(outer_elem, "coordinates"), None)
        if coords_elem is not None:
            outer = close_ring(parse_coord_text(coords_elem.text or ""))
            break

    for inner_elem in iter_children_by_name(poly_elem, "innerBoundaryIs"):
        coords_elem = next(iter_children_by_name(inner_elem, "coordinates"), None)
        if coords_elem is not None:
            hole = close_ring(parse_coord_text(coords_elem.text or ""))
            if len(hole) >= 4:
                holes.append(hole)

    if outer is None or len(outer) < 4:
        return None

    try:
        geom = Polygon(outer, holes)
        geom = make_valid(geom)
        if geom.is_empty:
            return None
        if geom.geom_type == "Polygon":
            return geom
        if geom.geom_type == "MultiPolygon":
            return geom
        # Sometimes make_valid returns GeometryCollection.
        polys = [g for g in getattr(geom, "geoms", []) if g.geom_type in ("Polygon", "MultiPolygon")]
        if polys:
            return unary_union(polys)
    except Exception:
        return None
    return None


def extract_kml_from_kmz(kmz_path: Path, tmp_dir: Path) -> Path:
    with zipfile.ZipFile(kmz_path, "r") as zf:
        kml_files = [n for n in zf.namelist() if n.lower().endswith(".kml")]
        if not kml_files:
            raise ValueError(f"No .kml file found inside KMZ: {kmz_path}")
        # Usually doc.kml is the main one. Prefer it if present.
        kml_name = "doc.kml" if "doc.kml" in kml_files else kml_files[0]
        zf.extract(kml_name, tmp_dir)
        return tmp_dir / kml_name


def parse_kml_file(kml_path: Path, source_file: Path):
    tree = ET.parse(kml_path)
    root = tree.getroot()
    rows = []

    placemarks = [elem for elem in root.iter() if strip_namespace(elem.tag) == "Placemark"]
    if not placemarks:
        print(f"Warning: no Placemark found in {source_file}")

    poly_counter = 0
    for pm_idx, pm in enumerate(placemarks, start=1):
        pm_name = child_text(pm, "name") or f"placemark_{pm_idx}"
        poly_elems = [elem for elem in pm.iter() if strip_namespace(elem.tag) == "Polygon"]
        for poly_idx, poly_elem in enumerate(poly_elems, start=1):
            geom = polygon_from_kml_polygon(poly_elem)
            if geom is None or geom.is_empty:
                continue
            # Explode multipolygons now for cleaner shapefiles.
            geoms = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            for part_idx, part in enumerate(geoms, start=1):
                if part is None or part.is_empty:
                    continue
                poly_counter += 1
                rows.append({
                    "source_file": source_file.name,
                    "source_stem": source_file.stem,
                    "pm_name": pm_name,
                    "pm_index": pm_idx,
                    "poly_index": poly_idx,
                    "part_id": part_idx,
                    "feature_id": poly_counter,
                    "geometry": part,
                })

    return rows


def read_kmz_or_kml(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".kmz":
        with tempfile.TemporaryDirectory() as td:
            kml_path = extract_kml_from_kmz(path, Path(td))
            rows = parse_kml_file(kml_path, path)
    elif suffix == ".kml":
        rows = parse_kml_file(path, path)
    else:
        rows = []
    return rows


def write_shapefile(gdf: gpd.GeoDataFrame, out_shp: Path):
    out_shp.parent.mkdir(parents=True, exist_ok=True)

    # Shapefile column names are limited; use short aliases.
    out = gdf.copy()
    rename = {
        "source_file": "src_file",
        "source_stem": "src_stem",
        "feature_id": "feat_id",
        "poly_index": "poly_idx",
    }
    out = out.rename(columns=rename)

    # Remove any stale shapefile sidecars before writing.
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]:
        p = out_shp.with_suffix(ext)
        if p.exists():
            p.unlink()

    out.to_file(out_shp, driver="ESRI Shapefile")


def write_gpkg(gdf: gpd.GeoDataFrame, out_gpkg: Path, layer: str):
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    if out_gpkg.exists():
        out_gpkg.unlink()
    gdf.to_file(out_gpkg, layer=layer, driver="GPKG", index=False)


def make_config_yaml(field_name: str, field_shp: Path, output_dir: Path, config_path: Path):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    text = f'''# Auto-generated example config for multimodal_eo_timeseries repository
project:
  field_name: "{field_name}"
  field_shapefile: "{field_shp}"
  output_dir: "{output_dir}"

gee:
  project: "your-earth-engine-project-id"

processing:
  start_year: 2020
  end_year: null
  season_start: "03-01"
  season_end: "10-30"

sentinel2:
  step_days: 5
  main_cloud_filter: 40
  fallback_cloud_filter: 95

sentinel1:
  step_days: 5
  orbit: null
  apply_speckle_filter: false

weather:
  start_year: 2000
'''
    config_path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Batch convert KMZ/KML files to field_shapefile.shp outputs.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, required=True, help="Directory containing KMZ/KML files.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, required=True, help="Output root directory.")
    parser.add_argument("--repo-folders", action="store_true", help="Create <field>/Satellite/Shapefile/field_shapefile.shp structure.")
    parser.add_argument("--merge-placemarks-per-file", action="store_true", help="Union all polygons in each KMZ/KML into one feature.")
    parser.add_argument("--write-gpkg", action="store_true", help="Also write GeoPackage outputs.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = sorted(list(input_dir.glob("*.kmz")) + list(input_dir.glob("*.kml")))
    if not files:
        raise FileNotFoundError(f"No KMZ/KML files found in: {input_dir}")

    all_gdfs = []
    report_rows = []

    print(f"Found {len(files)} KMZ/KML files in: {input_dir}")

    for file_path in files:
        print("=" * 80)
        print(f"Reading: {file_path.name}")
        rows = read_kmz_or_kml(file_path)
        if not rows:
            print(f"  No polygon features extracted from {file_path.name}")
            report_rows.append({
                "source_file": file_path.name,
                "status": "no_polygons",
                "n_features": 0,
                "output_shp": "",
            })
            continue

        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()

        if args.merge_placemarks_per_file:
            merged_geom = unary_union(list(gdf.geometry))
            gdf = gpd.GeoDataFrame([{
                "source_file": file_path.name,
                "source_stem": file_path.stem,
                "pm_name": file_path.stem,
                "pm_index": 1,
                "poly_index": 1,
                "part_id": 1,
                "feature_id": 1,
                "geometry": merged_geom,
            }], geometry="geometry", crs="EPSG:4326")

        field_name = safe_name(file_path.stem)
        gdf["field_name"] = field_name
        gdf["area_ha"] = gdf.to_crs(epsg=3857).area / 10000.0

        if args.repo_folders:
            shp_dir = output_root / field_name / "Satellite" / "Shapefile"
            sat_data_dir = output_root / field_name / "Satellite" / "Satellite_Data"
            out_shp = shp_dir / "field_shapefile.shp"
            make_config_yaml(
                field_name=field_name,
                field_shp=out_shp,
                output_dir=sat_data_dir,
                config_path=output_root / field_name / "config.yaml",
            )
        else:
            out_shp = output_root / field_name / "field_shapefile.shp"
            sat_data_dir = output_root / field_name / "Satellite_Data"

        write_shapefile(gdf, out_shp)
        print(f"  Saved shapefile: {out_shp}")

        if args.write_gpkg:
            out_gpkg = out_shp.parent / "field_shapefile.gpkg"
            write_gpkg(gdf, out_gpkg, layer="field_shapefile")
            print(f"  Saved GeoPackage: {out_gpkg}")

        all_gdfs.append(gdf)
        report_rows.append({
            "source_file": file_path.name,
            "field_name": field_name,
            "status": "ok",
            "n_features": len(gdf),
            "total_area_ha": float(gdf["area_ha"].sum()),
            "output_shp": str(out_shp),
            "suggested_output_dir": str(sat_data_dir),
        })

    report = pd.DataFrame(report_rows)
    report_csv = output_root / "kmz_to_shapefile_conversion_report.csv"
    report.to_csv(report_csv, index=False)
    print("=" * 80)
    print(f"Saved conversion report: {report_csv}")

    if all_gdfs:
        combined = pd.concat(all_gdfs, ignore_index=True)
        combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
        combined_shp = output_root / "all_kmz_polygons_combined.shp"
        write_shapefile(combined, combined_shp)
        print(f"Saved combined shapefile: {combined_shp}")

        if args.write_gpkg:
            combined_gpkg = output_root / "all_kmz_polygons_combined.gpkg"
            write_gpkg(combined, combined_gpkg, layer="all_kmz_polygons")
            print(f"Saved combined GeoPackage: {combined_gpkg}")

    print("Done. Hare Krishna.")


if __name__ == "__main__":
    main()
