#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:33:54 2026

@author: ashutosh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Faster streaming p01 for {FIELD_NAME}.

Key idea:
- For each year and each 5-day interval, select the best Sentinel-2 scene
  and download immediately, instead of scanning all intervals first.

Seasonal window:
- 2020 onward
- March 1 to October 30 each year

Outputs:
- cleaned polygons shapefile
- QA plot
- Sentinel-2 images
- Sentinel-1 images
- Sentinel-2 interval selection summary CSV
- per-polygon Sentinel-2 index CSVs
- combined Sentinel-2 CSV
"""

import os
import json
import requests
from datetime import datetime, timedelta

import ee
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, MultiPolygon, Point, LineString, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import make_valid


# =========================
# USER SETTINGS FROM config/config.yaml
# =========================
from eo_config import (
    FIELD_NAME, FIELD_SHP, OUTPUT_DIR, EE_PROJECT, START_YEAR, END_YEAR,
    SEASON_START_MONTH, SEASON_START_DAY, SEASON_END_MONTH, SEASON_END_DAY,
    S2_STEP_DAYS, S2_MAIN_CLOUD_FILTER, S2_FALLBACK_CLOUD_FILTER,
    S1_STEP_DAYS, S1_ORBIT, INNER_BUFFER_METERS,
    RUN_SENTINEL2_SELECTION_AND_DOWNLOAD, RUN_SENTINEL1_DOWNLOAD,
    RUN_S2_POLYGON_TIMESERIES, APPLY_CROP_MASK, CROP_MASK_SHP,
    print_config_summary, cleaned_shapefile_path
)

print_config_summary()
print(f"USING FASTER STREAMING P01 FOR {FIELD_NAME}")


# =========================
# HELPERS
# =========================
def strip_z(geom):
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == 'Polygon':
        return Polygon([(x, y) for x, y, *_ in geom.exterior.coords],
                       [[(x, y) for x, y, *_ in ring.coords] for ring in geom.interiors])
    elif geom.geom_type == 'MultiPolygon':
        return MultiPolygon([strip_z(part) for part in geom.geoms])
    elif geom.geom_type == 'Point':
        x, y = geom.coords[0][:2]
        return Point(x, y)
    elif geom.geom_type == 'LineString':
        return LineString([(x, y) for x, y, *_ in geom.coords])
    elif geom.geom_type == 'GeometryCollection':
        return GeometryCollection([strip_z(g) for g in geom.geoms if g is not None and not g.is_empty])
    return geom


def pick_name_column(gdf):
    for col in ['Name', 'NAME', 'name', 'Field', 'FIELD', 'field']:
        if col in gdf.columns:
            return col
    return None


def to_wgs84(gdf):
    if gdf.crs is None:
        raise ValueError('Input shapefile has no CRS defined.')
    return gdf.to_crs(epsg=4326)


def polygonal_only(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type in ('Polygon', 'MultiPolygon'):
        return geom
    if geom.geom_type == 'GeometryCollection':
        polys = [g for g in geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon') and not g.is_empty]
        if not polys:
            return None
        return unary_union(polys)
    return None


def repair_geometry(geom):
    if geom is None or geom.is_empty:
        return None
    geom = strip_z(geom)
    geom = make_valid(geom)
    geom = polygonal_only(geom)
    if geom is None or geom.is_empty:
        return None
    try:
        geom2 = geom.buffer(0)
        if geom2 is not None and not geom2.is_empty:
            geom = geom2
    except Exception:
        pass
    return geom if geom is not None and not geom.is_empty else None


def clean_gdf_geometries(gdf, label='layer'):
    gdf = gdf.copy()
    before = len(gdf)
    gdf['geometry'] = gdf['geometry'].apply(repair_geometry)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    after = len(gdf)
    print(f'{label}: kept {after}/{before} features after geometry repair.')
    if after == 0:
        raise ValueError(f'No valid polygon geometries remain in {label}.')
    return gdf


def _polygon_parts(geom):
    """Return polygonal parts after geometry repair."""
    geom = repair_geometry(geom)
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == 'Polygon':
        return [geom]
    if geom.geom_type == 'MultiPolygon':
        return [g for g in geom.geoms if g is not None and not g.is_empty]
    poly = polygonal_only(geom)
    if poly is None or poly.is_empty:
        return []
    if poly.geom_type == 'Polygon':
        return [poly]
    return [g for g in poly.geoms if g is not None and not g.is_empty]


def load_crop_mask_union(mask_shp):
    """Read an optional crop mask shapefile and return a single WGS84 geometry."""
    if mask_shp is None:
        return None
    if not os.path.exists(mask_shp):
        raise FileNotFoundError(f"Crop mask shapefile not found:\n{mask_shp}")

    mask_gdf = gpd.read_file(mask_shp)
    mask_gdf = to_wgs84(mask_gdf)
    mask_gdf = clean_gdf_geometries(mask_gdf, label='Crop mask shapefile')

    mask_geom = unary_union(list(mask_gdf.geometry))
    mask_geom = repair_geometry(mask_geom)
    if mask_geom is None or mask_geom.is_empty:
        raise ValueError(f"Crop mask has no valid polygon area after repair:\n{mask_shp}")
    return mask_geom


def apply_crop_mask_to_cleaned_polygons(cleaned_gdf, mask_shp):
    """Intersect cleaned field polygons with a crop mask.

    The output remains compatible with the rest of the p01-p10 workflow:
    it keeps zone_id, source_name, part_id, and geometry columns. The saved
    cleaned shapefile becomes the masked crop area used by p06/p07/etc.
    """
    if not APPLY_CROP_MASK or mask_shp is None:
        print('Crop mask: not used. Processing full field polygon(s).')
        return cleaned_gdf

    print(f'Crop mask: applying mask from {mask_shp}')
    mask_geom = load_crop_mask_union(mask_shp)

    out_rows = []
    for _, row in cleaned_gdf.iterrows():
        intersection = repair_geometry(row.geometry.intersection(mask_geom))
        parts = _polygon_parts(intersection)
        if not parts:
            continue
        for new_part_id, part in enumerate(parts, start=1):
            out_rows.append({
                'zone_id': int(row['zone_id']),
                'source_name': row.get('source_name', f'{FIELD_NAME}_Field_{int(row["zone_id"])}'),
                'part_id': new_part_id,
                'original_part_id': int(row.get('part_id', new_part_id)),
                'mask_applied': 1,
                'geometry': part,
            })

    if not out_rows:
        raise ValueError(
            'The crop mask did not overlap the input field shapefile. '\
            'Please check CRS, file paths, and whether the mask is in the field.'
        )

    masked_gdf = gpd.GeoDataFrame(out_rows, crs='EPSG:4326')
    print(f'Crop mask: retained {len(masked_gdf)} polygon part(s) after intersection.')
    return masked_gdf



def shapely_to_ee_geometry(geom):
    geojson_geom = json.loads(
        gpd.GeoSeries([geom], crs='EPSG:4326').to_json()
    )['features'][0]['geometry']
    return ee.Geometry(geojson_geom)


def safe_inner_buffer(geom, meters=None):
    if meters is None or meters == 0:
        return geom
    gdf = gpd.GeoDataFrame({'id': [1]}, geometry=[geom], crs='EPSG:4326').to_crs(epsg=3857)
    buffered = gdf.geometry.iloc[0].buffer(-meters)
    if buffered.is_empty:
        print(f'Warning: inward buffer of {meters} m removed the polygon. Using unbuffered polygon.')
        buffered = gdf.geometry.iloc[0]
    return gpd.GeoDataFrame({'id': [1]}, geometry=[buffered], crs='EPSG:3857').to_crs(epsg=4326).geometry.iloc[0]


def init_ee():
    try:
        ee.Initialize(project=EE_PROJECT) if EE_PROJECT else ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT) if EE_PROJECT else ee.Initialize()


def make_safe_name(text):
    return ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in str(text))


def save_labeled_plot(gdf, out_png):
    fig, ax = plt.subplots(figsize=(9, 9))
    gdf.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=1.5)

    for _, row in gdf.iterrows():
        pt = row.geometry.representative_point()
        label = f"zone {int(row['zone_id'])}\npart {int(row['part_id'])}"
        if 'source_name' in row and pd.notna(row['source_name']):
            label += f"\n{row['source_name']}"
        ax.text(
            pt.x, pt.y, label, fontsize=8, ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5)
        )

    ax.set_title(f'{FIELD_NAME} cleaned polygons')
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()


def build_clean_polygons(field_shp):
    field_gdf = gpd.read_file(field_shp)
    field_gdf = to_wgs84(field_gdf)
    field_gdf = clean_gdf_geometries(field_gdf, label='Field shapefile')

    name_col = pick_name_column(field_gdf)

    cleaned_rows = []
    zone_counter = 1

    for _, row in field_gdf.iterrows():
        if name_col is not None:
            source_name = str(row[name_col]).strip()
            if source_name == '':
                source_name = f'{FIELD_NAME}_Field_{zone_counter}'
        else:
            source_name = f'{FIELD_NAME}_Field_{zone_counter}'

        geom = repair_geometry(row.geometry)
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == 'Polygon':
            parts = [geom]
        elif geom.geom_type == 'MultiPolygon':
            parts = list(geom.geoms)
        else:
            poly = polygonal_only(geom)
            if poly is None:
                continue
            if poly.geom_type == 'Polygon':
                parts = [poly]
            else:
                parts = list(poly.geoms)

        for part_id, part in enumerate(parts, start=1):
            part = repair_geometry(part)
            if part is None or part.is_empty:
                continue
            cleaned_rows.append({
                'zone_id': zone_counter,
                'source_name': source_name,
                'part_id': part_id,
                'geometry': part
            })
        zone_counter += 1

    if not cleaned_rows:
        raise ValueError('No valid polygons remain after geometry cleaning.')

    return gpd.GeoDataFrame(cleaned_rows, crs='EPSG:4326')


def save_cleaned_shapefile(cleaned_gdf, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    shp_path = cleaned_shapefile_path()
    cleaned_gdf.to_file(shp_path)
    return shp_path


def plot_cleaned_polygons(cleaned_gdf, output_dir):
    plot_dir = os.path.join(output_dir, 'QA_QC')
    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    cleaned_gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1.5)

    for _, row in cleaned_gdf.iterrows():
        x, y = row.geometry.representative_point().coords[0]
        label = f"zone {row['zone_id']}\npart {row['part_id']}"
        ax.text(x, y, label, fontsize=9, ha='center', va='center')

    ax.set_title(f'Cleaned {FIELD_NAME} Polygons')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')

    out_png = os.path.join(plot_dir, f'{FIELD_NAME}_cleaned_polygons_labeled.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved polygon QA plot: {out_png}')


def get_season_ranges(start_year, end_year):
    ranges = []
    current_year = datetime.now().year
    current_date = datetime.now().date()

    for year in range(start_year, end_year + 1):
        season_start = datetime(year, SEASON_START_MONTH, SEASON_START_DAY)
        season_end = datetime(year, SEASON_END_MONTH, SEASON_END_DAY)

        if year == current_year and season_end.date() > current_date:
            season_end = datetime.combine(current_date, datetime.min.time())

        if season_start < season_end:
            ranges.append((season_start.strftime('%Y-%m-%d'),
                           season_end.strftime('%Y-%m-%d'),
                           year))
    return ranges


def generate_intervals(start_date, end_date, step_days=5):
    intervals = []
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    while current < end:
        nxt = min(current + timedelta(days=step_days), end)
        intervals.append((current.strftime('%Y-%m-%d'), nxt.strftime('%Y-%m-%d')))
        current = nxt
    return intervals


def download_ee_image(image, bands, file_path, ee_aoi, scale, crs='EPSG:4326'):
    image = ee.Image(image)
    try:
        url = image.clip(ee_aoi).select(bands).getDownloadURL({
            'scale': scale,
            'region': ee_aoi.getInfo(),
            'format': 'GEO_TIFF',
            'crs': crs
        })
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f'Downloaded: {file_path}')
    except Exception as e:
        print(f'Failed: {file_path} -> {e}')


def calculate_ndvi(image):
    image = ee.Image(image)
    return image.addBands(image.normalizedDifference(['B8', 'B4']).rename('NDVI'))


def calculate_savi(image, L=0.5):
    image = ee.Image(image)
    savi = image.expression(
        '((NIR - RED) / (NIR + RED + L)) * (1 + L)',
        {'NIR': image.select('B8'), 'RED': image.select('B4'), 'L': L}
    ).rename('SAVI')
    return image.addBands(savi)


def calculate_ndwi(image):
    image = ee.Image(image)
    return image.addBands(image.normalizedDifference(['B3', 'B8']).rename('NDWI'))


def calculate_ndre(image):
    image = ee.Image(image)
    return image.addBands(image.normalizedDifference(['B8', 'B5']).rename('NDRE'))


def extract_time_series(image_collection, band_name, ee_aoi, scale=10):
    def extract_feature(image):
        image = ee.Image(image)
        date = image.date().format('YYYY-MM-dd')
        mean_value = image.select(band_name).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee_aoi,
            scale=scale,
            maxPixels=1e13
        ).get(band_name)
        cloud_pct = ee.Algorithms.If(
            image.propertyNames().contains('CLOUDY_PIXEL_PERCENTAGE'),
            image.get('CLOUDY_PIXEL_PERCENTAGE'),
            None
        )
        return ee.Feature(None, {'date': date, band_name: mean_value, 'cloud_pct': cloud_pct})

    time_series = image_collection.map(extract_feature).getInfo()
    features = time_series['features']
    rows = [
        {
            'date': f['properties']['date'],
            band_name: f['properties'].get(band_name),
            'cloud_pct': f['properties'].get('cloud_pct')
        }
        for f in features if f['properties'].get(band_name) is not None
    ]
    return pd.DataFrame(rows)


def build_s2_collection(ee_aoi, start_date, end_date, cloud_filter):
    return (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start_date, end_date)
        .filterBounds(ee_aoi)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_filter))
        .sort('CLOUDY_PIXEL_PERCENTAGE')
    )


def choose_best_s2_image_for_interval(ee_aoi, start, end):
    main_subset = build_s2_collection(ee_aoi, start, end, S2_MAIN_CLOUD_FILTER)
    n_main = main_subset.size().getInfo()

    if n_main > 0:
        img = ee.Image(main_subset.first())
        return img, f"main<= {S2_MAIN_CLOUD_FILTER}"

    fallback_subset = build_s2_collection(ee_aoi, start, end, S2_FALLBACK_CLOUD_FILTER)
    n_fb = fallback_subset.size().getInfo()

    if n_fb > 0:
        img = ee.Image(fallback_subset.first())
        return img, f"fallback<= {S2_FALLBACK_CLOUD_FILTER}"

    return None, "none"


def download_sentinel2_streaming(ee_aoi, output_dir):
    s2_dir = os.path.join(output_dir, 'Sentinel2', 'images')
    os.makedirs(s2_dir, exist_ok=True)

    summary_rows = []
    selected_images = []

    total_selected = 0
    total_downloaded = 0
    total_existing = 0

    season_ranges = get_season_ranges(START_YEAR, END_YEAR)
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

    for season_start, season_end, year in season_ranges:
        print(f"\nProcessing Sentinel-2 for year {year}: {season_start} to {season_end}")
        intervals = generate_intervals(season_start, season_end, step_days=S2_STEP_DAYS)

        for start, end in intervals:
            img, mode = choose_best_s2_image_for_interval(ee_aoi, start, end)

            if img is None:
                summary_rows.append({
                    'year': year,
                    'interval_start': start,
                    'interval_end': end,
                    'selected': 0,
                    'selected_date': None,
                    'cloud_pct': None,
                    'mode': mode
                })
                print(f"  {start} to {end}: no S2 image found")
                continue

            date = img.date().format('YYYY-MM-dd').getInfo()
            cloud_pct = img.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
            out_tif = os.path.join(s2_dir, f'Sentinel2_{date}.tif')

            summary_rows.append({
                'year': year,
                'interval_start': start,
                'interval_end': end,
                'selected': 1,
                'selected_date': date,
                'cloud_pct': cloud_pct,
                'mode': mode
            })

            total_selected += 1
            selected_images.append(img)

            if os.path.exists(out_tif):
                total_existing += 1
                print(f"  {start} to {end}: selected {date} (cloud {cloud_pct}%) already exists")
            else:
                print(f"  {start} to {end}: downloading {date} (cloud {cloud_pct}%) [{mode}]")
                download_ee_image(img, bands, out_tif, ee_aoi, scale=10)
                total_downloaded += 1

    sel_csv = os.path.join(output_dir, 'Sentinel2', 's2_interval_selection_summary.csv')
    pd.DataFrame(summary_rows).to_csv(sel_csv, index=False)
    print(f"\nSaved S2 interval selection summary: {sel_csv}")
    print(f"Sentinel-2 summary: selected={total_selected}, downloaded={total_downloaded}, existing={total_existing}")

    if len(selected_images) == 0:
        raise ValueError("No Sentinel-2 images selected.")

    selected_ic = ee.ImageCollection.fromImages(selected_images)
    s2_with_idx = selected_ic.map(calculate_ndvi).map(calculate_savi).map(calculate_ndwi).map(calculate_ndre)
    return s2_with_idx


def download_sentinel1(ee_aoi, output_dir, orbit_pass=None):
    s1_dir = os.path.join(output_dir, 'Sentinel1', 'images')
    os.makedirs(s1_dir, exist_ok=True)

    season_ranges = get_season_ranges(START_YEAR, END_YEAR)
    bands = ['VV', 'VH', 'angle']

    for season_start, season_end, year in season_ranges:
        print(f"\nProcessing Sentinel-1 for year {year}: {season_start} to {season_end}")

        s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
              .filterDate(season_start, season_end)
              .filterBounds(ee_aoi)
              .filter(ee.Filter.eq('instrumentMode', 'IW'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))

        if orbit_pass in ['ASCENDING', 'DESCENDING']:
            s1 = s1.filter(ee.Filter.eq('orbitProperties_pass', orbit_pass))

        intervals = generate_intervals(season_start, season_end, step_days=S1_STEP_DAYS)

        for start, end in intervals:
            subset = s1.filterDate(start, end)
            n = subset.size().getInfo()
            if n == 0:
                continue

            img_list = subset.toList(n)
            for i in range(n):
                img = ee.Image(img_list.get(i))
                date = img.date().format('YYYY-MM-dd').getInfo()
                orbit = img.get('orbitProperties_pass').getInfo()
                out_tif = os.path.join(s1_dir, f'Sentinel1_{orbit}_{date}.tif')

                if not os.path.exists(out_tif):
                    print(f"  {start} to {end}: downloading S1 {orbit} {date}")
                    download_ee_image(img, bands, out_tif, ee_aoi, scale=10)


def save_per_polygon_time_series(s2_with_idx, cleaned_gdf, output_dir, inner_buffer_meters=None):
    """
    Extract per-polygon Sentinel-2 index time series.

    Important fix:
    - extract_time_series() returns cloud_pct with every index table.
    - If cloud_pct is merged repeatedly with NDVI/SAVI/NDWI/NDRE, pandas creates
      cloud_pct_x/cloud_pct_y and eventually raises:
      MergeError: Passing 'suffixes' which cause duplicate columns {'cloud_pct_x'}.
    - This version merges only the index-value columns first, then adds one
      cloud_pct column once at the end.
    """
    ts_dir = os.path.join(output_dir, 'Sentinel2', 'per_polygon_time_series')
    os.makedirs(ts_dir, exist_ok=True)

    all_rows = []

    for _, row in cleaned_gdf.iterrows():
        zone_id = row['zone_id']
        source_name = row['source_name']
        part_id = row['part_id']
        geom = row.geometry

        geom_for_stats = safe_inner_buffer(geom, inner_buffer_meters)
        ee_geom = shapely_to_ee_geometry(geom_for_stats)

        index_specs = [
            (extract_time_series(s2_with_idx, 'NDVI', ee_geom), 'NDVI'),
            (extract_time_series(s2_with_idx, 'SAVI', ee_geom), 'SAVI'),
            (extract_time_series(s2_with_idx, 'NDWI', ee_geom), 'NDWI'),
            (extract_time_series(s2_with_idx, 'NDRE', ee_geom), 'NDRE'),
        ]

        merged = None
        cloud_df = None

        for df_idx, col in index_specs:
            if df_idx is None or df_idx.empty:
                continue

            df_idx = df_idx.copy()
            df_idx['date'] = pd.to_datetime(df_idx['date'], errors='coerce').dt.floor('D')
            df_idx = df_idx.dropna(subset=['date']).copy()

            # Keep each index column only once. Duplicate dates can occur if the same
            # scene is selected more than once, so keep the first record per date.
            value_df = (
                df_idx[['date', col]]
                .drop_duplicates(subset=['date'], keep='first')
                .copy()
            )

            if merged is None:
                merged = value_df
            else:
                merged = merged.merge(value_df, on='date', how='outer')

            # Save cloud_pct once only. It is image-level metadata, not index-specific.
            if cloud_df is None and 'cloud_pct' in df_idx.columns:
                cloud_df = (
                    df_idx[['date', 'cloud_pct']]
                    .drop_duplicates(subset=['date'], keep='first')
                    .copy()
                )

        if merged is not None and not merged.empty:
            if cloud_df is not None and not cloud_df.empty:
                merged = merged.merge(cloud_df, on='date', how='left')

            merged['zone_id'] = zone_id
            merged['source_name'] = source_name
            merged['part_id'] = part_id

            out_cols = ['zone_id', 'source_name', 'part_id', 'date', 'NDVI', 'SAVI', 'NDWI', 'NDRE']
            if 'cloud_pct' in merged.columns:
                out_cols.append('cloud_pct')

            # Some indices can be missing if Earth Engine returned no value for a date.
            # Add missing columns as NA so downstream p02/p02.1 still receive the same schema.
            for c in out_cols:
                if c not in merged.columns:
                    merged[c] = pd.NA

            merged = merged[out_cols].sort_values('date')

            safe_name = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in str(source_name))
            out_csv = os.path.join(ts_dir, f'zone_{int(zone_id):02d}_{safe_name}_part_{int(part_id)}_indices_time_series.csv')
            merged.to_csv(out_csv, index=False)
            print(f'Saved per-polygon time series: {out_csv}')

            all_rows.append(merged)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True).sort_values(['zone_id', 'part_id', 'date'])
        combined_csv = os.path.join(ts_dir, 'all_polygons_indices_time_series.csv')
        combined.to_csv(combined_csv, index=False)
        print(f'Saved combined time series: {combined_csv}')
    else:
        print('Warning: no per-polygon Sentinel-2 time-series rows were created.')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_ee()

    cleaned_gdf = build_clean_polygons(FIELD_SHP)
    cleaned_gdf = apply_crop_mask_to_cleaned_polygons(cleaned_gdf, CROP_MASK_SHP)
    cleaned_shp = save_cleaned_shapefile(cleaned_gdf, OUTPUT_DIR)
    print(f'Cleaned polygons shapefile saved to: {cleaned_shp}')

    plot_cleaned_polygons(cleaned_gdf, OUTPUT_DIR)

    merged_aoi = unary_union(list(cleaned_gdf.geometry))
    ee_aoi = shapely_to_ee_geometry(merged_aoi)

    s2_with_idx = None
    if RUN_SENTINEL2_SELECTION_AND_DOWNLOAD or RUN_S2_POLYGON_TIMESERIES:
        print('Selecting/downloading Sentinel-2...')
        s2_with_idx = download_sentinel2_streaming(ee_aoi, OUTPUT_DIR)
    else:
        print('Skipping Sentinel-2 selection/download.')

    if RUN_SENTINEL1_DOWNLOAD:
        print('\nDownloading Sentinel-1...')
        download_sentinel1(ee_aoi, OUTPUT_DIR, orbit_pass=S1_ORBIT)
    else:
        print('\nSkipping Sentinel-1 download because RUN_SENTINEL1_DOWNLOAD = False.')

    if RUN_S2_POLYGON_TIMESERIES:
        if s2_with_idx is None:
            raise RuntimeError('RUN_S2_POLYGON_TIMESERIES=True requires Sentinel-2 selection to build s2_with_idx.')
        print('\nSaving per-polygon Sentinel-2 index time series...')
        save_per_polygon_time_series(s2_with_idx, cleaned_gdf, OUTPUT_DIR, inner_buffer_meters=INNER_BUFFER_METERS)
    else:
        print('Skipping per-polygon Sentinel-2 time-series extraction.')

    print('Done. Hare Krishna.')


if __name__ == '__main__':
    main()
