# Script summary

| Script | Purpose |
|---|---|
| `prepare_config.py` | Creates `config/config.yaml` for a new field. |
| `kmz_to_shapefile_batch.py` | Converts KMZ/KML field boundaries into shapefiles. |
| `p01_s1_s2_download_fast.py` | Cleans the field boundary, optionally applies a crop mask, downloads Sentinel-2 and Sentinel-1 imagery, and creates Sentinel-2 index time series. |
| `p02_remove_outliers_NDVI.py` | Cleans NDVI time series. |
| `p02_1_remove_outliers_all_indices_by_ndvi.py` | Applies NDVI-based quality control to additional Sentinel-2 indices. |
| `p03_plot_yearwise_indices_with_zone_mean.py` | Creates yearly Sentinel-2 index plots. |
| `p04_make_zonewise_pixelwise_ndvi_spaghetti.py` | Creates pixelwise NDVI spaghetti plots. |
| `p05_download_weather_timeseries.py` | Downloads daily gridMET, CHIRPS, and ERA5-Land time series. |
| `p06_sentinel1_zone_timeseries.py` | Extracts zone-level Sentinel-1 VV/VH time series. |
| `p07_sentinel1_clean_plot_v3.py` | Cleans and plots Sentinel-1 time series. |
| `p06a_landsat_download_average_ndvi.py` | Creates Landsat NDVI time series. |
| `p06b_landsat_remove_outliers_per_polygon.py` | Cleans Landsat NDVI time series. |
| `p06c_landsat_plot_ndvi_per_polygon_yearwise.py` | Plots Landsat NDVI time series by year. |
| `p08_build_multimodal_table.py` | Merges Sentinel-2, Sentinel-1, weather, and Landsat outputs. |
| `p09_multimodal_yearly_plot_full_context.py` | Creates full-context yearly multimodal plots. |
| `p10_multimodal_twofigures_4panels.py` | Creates summary multimodal panel figures. |
