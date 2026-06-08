# Quick start for a new field

## 1. Prepare a shapefile

The repository expects a polygon shapefile, usually named:

```text
field_shapefile.shp
```

The shapefile must include the `.shp`, `.shx`, `.dbf`, and `.prj` components.

## 2. Create a config file

From the repository root:

```bash
python scripts/prepare_config.py \
  --field-name My_Field \
  --field-shapefile "/absolute/path/to/field_shapefile.shp" \
  --output-dir "/absolute/path/to/Satellite_Data" \
  --ee-project "your-earth-engine-project-id" \
  --write config/config.yaml
```

For Windows, forward slashes are recommended:

```text
C:/Users/Name/Documents/My_Field/field_shapefile.shp
```

## 3. Run the workflow

Run scripts one at a time:

```bash
python scripts/p01_s1_s2_download_fast.py
python scripts/p02_remove_outliers_NDVI.py
python scripts/p02_1_remove_outliers_all_indices_by_ndvi.py
python scripts/p03_plot_yearwise_indices_with_zone_mean.py
python scripts/p04_make_zonewise_pixelwise_ndvi_spaghetti.py
python scripts/p05_download_weather_timeseries.py
python scripts/p06_sentinel1_zone_timeseries.py
python scripts/p07_sentinel1_clean_plot_v3.py
python scripts/p06a_landsat_download_average_ndvi.py
python scripts/p06b_landsat_remove_outliers_per_polygon.py
python scripts/p06c_landsat_plot_ndvi_per_polygon_yearwise.py
python scripts/p08_build_multimodal_table.py
python scripts/p09_multimodal_yearly_plot_full_context.py
python scripts/p10_multimodal_twofigures_4panels.py
```

## 4. Expected output folders

```text
Satellite_Data/
├── Sentinel2/
├── Sentinel1/
├── Sentinel1_TimeSeries/
├── Landsat/
├── Weather_TimeSeries/
└── Multimodal_TimeSeries/
```
