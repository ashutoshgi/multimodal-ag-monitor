# Multimodal EO Time-Series Workflow for Crop Monitoring

This repository provides a configurable Python and Google Earth Engine workflow for downloading, cleaning, and plotting field-scale multimodal Earth observation time series for crop monitoring. A user supplies a field boundary shapefile, and the workflow generates Sentinel-2 vegetation-index time series, Sentinel-1 SAR backscatter time series, Landsat NDVI, and gridded weather variables for crop-growth analysis.

The workflow is intended for agricultural remote sensing, crop growth modeling, in-season monitoring, and downstream machine learning or deep learning applications.

---

## What this workflow downloads and processes

| Data source | Product type | Main variables produced | Why it helps crop monitoring |
|---|---|---|---|
| Sentinel-2 surface reflectance | Optical multispectral | NDVI, SAVI, NDWI, NDRE | Canopy greenness, vigor, water-related spectral response, and seasonal crop development |
| Sentinel-1 GRD SAR | C-band radar | VV, VH, VV−VH, VH−VV, VV/VH, VH/VV | Crop structure, moisture-sensitive response, cloud-independent monitoring |
| Landsat surface reflectance | Long-term optical | NDVI | Historical vegetation dynamics and long-term seasonal context |
| gridMET | Daily weather | Precipitation, reference ET, VPD | Crop water demand, atmospheric stress, and weather context |
| CHIRPS | Daily rainfall | Precipitation | Independent rainfall time series |
| ERA5-Land | Reanalysis | Temperature, precipitation, soil moisture-related variables | Environmental forcing and soil-water context |

---

## Repository structure

```text
multimodal-eo-timeseries/
├── README.md
├── LICENSE
├── CITATION.cff
├── AUTHORS.md
├── CHANGELOG.md
├── environment.yml
├── requirements.txt
├── run_all.sh
├── config/
│   ├── README.md
│   ├── config.example.yaml
│   └── config.with_crop_mask.example.yaml
├── docs/
│   ├── INSTALLATION.md
│   ├── QUICK_START_NEW_FIELD.md
│   ├── SCRIPT_SUMMARY.md
│   ├── CROP_MASK_USAGE.md
│   ├── DATASETS.md
│   ├── TROUBLESHOOTING.md
│   └── WINDOWS_SPYDER_NOTES.md
└── scripts/
    ├── prepare_config.py
    ├── kmz_to_shapefile_batch.py
    ├── p01_s1_s2_download_fast.py
    ├── p02_remove_outliers_NDVI.py
    ├── p02_1_remove_outliers_all_indices_by_ndvi.py
    ├── p03_plot_yearwise_indices_with_zone_mean.py
    ├── p04_make_zonewise_pixelwise_ndvi_spaghetti.py
    ├── p05_download_weather_timeseries.py
    ├── p06_sentinel1_zone_timeseries.py
    ├── p07_sentinel1_clean_plot_v3.py
    ├── p06a_landsat_download_average_ndvi.py
    ├── p06b_landsat_remove_outliers_per_polygon.py
    ├── p06c_landsat_plot_ndvi_per_polygon_yearwise.py
    ├── p08_build_multimodal_table.py
    ├── p09_multimodal_yearly_plot_full_context.py
    └── p10_multimodal_twofigures_4panels.py
```

---

## Installation

The recommended installation method is Conda/Mamba because geospatial packages depend on GDAL.

```bash
conda env create -f environment.yml
conda activate multimodal-eo
earthengine authenticate
```

If you prefer pip, create a Python environment and run:

```bash
pip install -r requirements.txt
earthengine authenticate
```

For Windows users, Conda/Mamba is strongly recommended. See `docs/WINDOWS_SPYDER_NOTES.md`.

---

## Quick start for a new field

From the repository root, create a config file:

```bash
python scripts/prepare_config.py   --field-name Example_Field   --field-shapefile "/absolute/path/to/field_shapefile.shp"   --output-dir "/absolute/path/to/Satellite_Data"   --ee-project "your-earth-engine-project-id"   --write config/config.yaml
```

Then run:

```bash
python scripts/p01_s1_s2_download_fast.py
```

After p01 finishes, run the processing scripts in order:

```bash
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

You can also run all scripts with:

```bash
bash run_all.sh
```

For a new user, running one script at a time is usually easier for debugging.

---

## Crop-mask option

If the field boundary contains roads, farmstead areas, ditches, or non-crop patches, you can use a crop mask shapefile. The p01 script will intersect the field boundary with the crop mask before downloading and summarizing data.

```bash
python scripts/prepare_config.py   --field-name Example_Field   --field-shapefile "/absolute/path/to/field_shapefile.shp"   --crop-mask-shapefile "/absolute/path/to/crop_mask.shp"   --output-dir "/absolute/path/to/Satellite_Data"   --ee-project "your-earth-engine-project-id"   --write config/config.yaml
```

See `docs/CROP_MASK_USAGE.md` for details.

---

## KMZ/KML conversion utility

If field boundaries are stored as KMZ/KML files, convert them to shapefiles first:

```bash
python scripts/kmz_to_shapefile_batch.py   --input-dir "/path/to/kmz_folder"   --output-root "/path/to/output_shapefiles"   --repo-folders   --write-gpkg
```

This creates `field_shapefile.shp` files that can be used with this repository.

---

## Citation

If you use this repository in research, teaching, reports, or derived workflows, please cite it. See `CITATION.cff` and the citation section below.

```text
Please cite the following if using the application:

Ashutosh Tiwari A, Mahendra Bhandari, Shweta Panjwani, Reshmi Sarkar, Rahul Raman, Sk Musfiq Us Salehin, Ram Ray, Mahendra Bhandari, Gurjindar Baath, Nithya Rajan, 2026. Multimodal EO Time-Series Workflow for Crop Monitoring. GitHub repository.

```

## License

This repository is released under the MIT License. See `LICENSE`.

---

