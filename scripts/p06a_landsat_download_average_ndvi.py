#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download stable Landsat Collection 2 Level 2 Tier 1 imagery (Landsat 4/5/7/8/9),
compute per-polygon mean NDVI, and save image GeoTIFFs plus NDVI CSVs.

Workflow:
2) Build a merged AOI for downloads
3) Query Landsat 4/5/7/8/9 C02 T1 L2 collections
4) Apply reflectance scaling and QA-based cloud/shadow/snow masking
5) Harmonize bands to common names: Blue, Green, Red, NIR, SWIR1, SWIR2
6) Download selected images (multiband reflectance)
7) Compute mean NDVI per polygon and save combined CSV + per-polygon CSVs

Hare Krishna
"""
#Hare Krishna

import os
import json
import requests
from datetime import datetime

import ee
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

# =========================
# USER SETTINGS
# =========================
#Hare Krishna
#Hare Krishna
#Hare Krishna
#Hare Krishna

BASE_DIR = OUTPUT_DIR
CLEANED_SHP = cleaned_shapefile_path()
OUTPUT_DIR = os.path.join(BASE_DIR, "Landsat")


# EE_PROJECT loaded from config
START_DATE = f'{LANDSAT_START_YEAR}-01-01'   # stable SR archive starts with Landsat 5 in 1984
if LANDSAT_END_YEAR >= datetime.now().year:
    END_DATE = datetime.now().strftime('%Y-%m-%d')
else:
    END_DATE = f'{LANDSAT_END_YEAR}-12-31'
MAX_CLOUD_COVER = 20         # scene-level filter; pixel-level QA mask still applied
DOWNLOAD_IMAGES = LANDSAT_DOWNLOAD_IMAGES #True
EXPORT_SCALE = 30
INNER_BUFFER_METERS = None   # e.g., 15 or None
KEEP_ONLY_GROWING_SEASON = False
GROWING_SEASON_MONTHS = [4, 5, 6, 7, 8, 9, 10]

# =========================
# HELPERS
# =========================
def detect_source_name_column(gdf):
    for col in ['source_name', 'source_nam', 'source', 'name', 'Name', 'NAME']:
        if col in gdf.columns:
            return col
    return None


def safe_inner_buffer(geom, meters=None):
    if meters is None or meters == 0:
        return geom
    gdf = gpd.GeoDataFrame({'id': [1]}, geometry=[geom], crs='EPSG:4326').to_crs(epsg=3857)
    buffered = gdf.geometry.iloc[0].buffer(-meters)
    if buffered.is_empty:
        print(f'Warning: inward buffer of {meters} m removed polygon. Using original polygon.')
        buffered = gdf.geometry.iloc[0]
    return gpd.GeoDataFrame({'id': [1]}, geometry=[buffered], crs='EPSG:3857').to_crs(epsg=4326).geometry.iloc[0]


def shapely_to_ee_geometry(geom):
    geojson_geom = json.loads(gpd.GeoSeries([geom], crs='EPSG:4326').to_json())['features'][0]['geometry']
    return ee.Geometry(geojson_geom)


def make_safe_name(text):
    return ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in str(text))


def download_ee_image(image, bands, file_path, ee_aoi, scale, crs='EPSG:4326'):
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


def prep_polygons(shp_path):
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError('Cleaned polygons shapefile is empty.')
    if gdf.crs is None:
        raise ValueError('Cleaned polygons shapefile has no CRS.')
    gdf = gdf.to_crs(epsg=4326)

    needed = ['zone_id', 'part_id', 'geometry']
    missing = [c for c in needed if c not in gdf.columns]
    if missing:
        raise ValueError(f'Missing required columns in cleaned shapefile: {missing}')

    source_col = detect_source_name_column(gdf)
    if source_col is None:
        gdf['source_label'] = gdf.apply(lambda r: f"zone_{int(r['zone_id']):02d}_part_{int(r['part_id'])}", axis=1)
    else:
        gdf['source_label'] = gdf[source_col].astype(str)
    return gdf


# -------------------------
# Landsat preprocessing
# -------------------------
def apply_scale_factors(image):
    optical = image.select('SR_B.*').multiply(2.75e-05).add(-0.2)
    return image.addBands(optical, overwrite=True)


def mask_landsat_c2(image):
    qa = image.select('QA_PIXEL')
    # Keep clear pixels: remove fill, dilated cloud, cirrus, cloud, cloud shadow, snow
    mask = (
        qa.bitwiseAnd(1 << 0).eq(0)
        .And(qa.bitwiseAnd(1 << 1).eq(0))
        .And(qa.bitwiseAnd(1 << 2).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
    )
    sat_mask = image.select('QA_RADSAT').eq(0)
    return image.updateMask(mask).updateMask(sat_mask)


def prep_l57(image):
    image = apply_scale_factors(image)
    image = mask_landsat_c2(image)
    renamed = image.select(
        ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
        ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
    )
    ndvi = renamed.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    return (renamed.addBands(ndvi)
            .copyProperties(image, image.propertyNames())
            .set('sensor_family', ee.String(image.get('SPACECRAFT_ID'))))


def prep_l89(image):
    image = apply_scale_factors(image)
    image = mask_landsat_c2(image)
    renamed = image.select(
        ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
        ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
    )
    ndvi = renamed.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    return (renamed.addBands(ndvi)
            .copyProperties(image, image.propertyNames())
            .set('sensor_family', ee.String(image.get('SPACECRAFT_ID'))))


def build_landsat_collection(ee_aoi, start_date, end_date, max_cloud_cover=80):
    l4 = (ee.ImageCollection('LANDSAT/LT04/C02/T1_L2')
          .filterBounds(ee_aoi)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lte('CLOUD_COVER', max_cloud_cover))
          .map(prep_l57))

    l5 = (ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
          .filterBounds(ee_aoi)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lte('CLOUD_COVER', max_cloud_cover))
          .map(prep_l57))

    l7 = (ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
          .filterBounds(ee_aoi)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lte('CLOUD_COVER', max_cloud_cover))
          .map(prep_l57))

    l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filterBounds(ee_aoi)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lte('CLOUD_COVER', max_cloud_cover))
          .map(prep_l89))

    l9 = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
          .filterBounds(ee_aoi)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lte('CLOUD_COVER', max_cloud_cover))
          .map(prep_l89))

    merged = l4.merge(l5).merge(l7).merge(l8).merge(l9).sort('system:time_start')
    if KEEP_ONLY_GROWING_SEASON:
        merged = merged.filter(ee.Filter.calendarRange(min(GROWING_SEASON_MONTHS), max(GROWING_SEASON_MONTHS), 'month'))
    return merged


def extract_ndvi_table(ic, ee_geom, zone_id, source_label, part_id):
    def one_feature(image):
        reduction = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee_geom,
            scale=30,
            maxPixels=1e13
        )
        return ee.Feature(None, {
            'date': image.date().format('YYYY-MM-dd'),
            'NDVI': reduction.get('NDVI'),
            'sensor_family': image.get('sensor_family'),
            'cloud_cover': image.get('CLOUD_COVER'),
            'landsat_product_id': image.get('LANDSAT_PRODUCT_ID'),
            'landsat_scene_id': image.get('LANDSAT_SCENE_ID'),
            'zone_id': zone_id,
            'source_name': source_label,
            'part_id': part_id,
        })

    fc = ic.map(one_feature).getInfo()['features']
    rows = []
    for feat in fc:
        props = feat['properties']
        if props.get('NDVI') is None:
            continue
        rows.append(props)
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_dir = os.path.join(OUTPUT_DIR, 'images')
    csv_dir = os.path.join(OUTPUT_DIR, 'per_polygon_time_series')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    try:
        ee.Initialize(project=EE_PROJECT) if EE_PROJECT else ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT) if EE_PROJECT else ee.Initialize()

    gdf = prep_polygons(CLEANED_SHP)
    merged_geom = unary_union(list(gdf.geometry))
    ee_aoi = shapely_to_ee_geometry(merged_geom)

    landsat = build_landsat_collection(ee_aoi, START_DATE, END_DATE, MAX_CLOUD_COVER)
    n_images = landsat.size().getInfo()
    print(f'Landsat images selected: {n_images}')

    if DOWNLOAD_IMAGES:
        bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI']
        img_list = landsat.toList(n_images)
        for i in range(n_images):
            img = ee.Image(img_list.get(i))
            date = img.date().format('YYYY-MM-dd').getInfo()
            pid = img.get('LANDSAT_PRODUCT_ID').getInfo()
            safe_pid = make_safe_name(pid)
            out_tif = os.path.join(image_dir, f'Landsat_{date}_{safe_pid}.tif')
            if not os.path.exists(out_tif):
                download_ee_image(img, bands, out_tif, ee_aoi, scale=EXPORT_SCALE)

    all_tables = []
    for _, row in gdf.iterrows():
        zone_id = int(row['zone_id'])
        part_id = int(row['part_id'])
        source_label = str(row['source_label'])
        geom = safe_inner_buffer(row.geometry, INNER_BUFFER_METERS)
        ee_geom = shapely_to_ee_geometry(geom)

        df = extract_ndvi_table(landsat, ee_geom, zone_id, source_label, part_id)
        if df.empty:
            print(f'No NDVI records for zone {zone_id}, part {part_id}')
            continue

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        all_tables.append(df)

        safe_label = make_safe_name(source_label)
        out_csv = os.path.join(csv_dir, f'zone_{zone_id:02d}_{safe_label}_part_{part_id}_landsat_ndvi.csv')
        df.to_csv(out_csv, index=False)
        print(f'Saved: {out_csv}')

    if all_tables:
        combined = pd.concat(all_tables, ignore_index=True).sort_values(['zone_id', 'part_id', 'date'])
        combined_csv = os.path.join(csv_dir, 'all_polygons_landsat_ndvi.csv')
        combined.to_csv(combined_csv, index=False)
        print(f'Saved combined CSV: {combined_csv}')

    print('Done. Hare Krishna.')


if __name__ == '__main__':
    main()
