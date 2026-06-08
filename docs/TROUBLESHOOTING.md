# Troubleshooting

## Config file not found

Create `config/config.yaml`:

```bash
cp config/config.example.yaml config/config.yaml
```

or generate it:

```bash
python scripts/prepare_config.py --field-shapefile "/path/to/field_shapefile.shp" --write config/config.yaml
```

## Earth Engine authentication error

Run:

```bash
earthengine authenticate
```

If your Earth Engine account requires a project ID, set it in `config/config.yaml`.

## Shapefile cannot be read

Make sure all shapefile sidecar files are present:

```text
field_shapefile.shp
field_shapefile.shx
field_shapefile.dbf
field_shapefile.prj
```

## No Sentinel-1 time-series CSV for p07

Run p06 before p07:

```bash
python scripts/p06_sentinel1_zone_timeseries.py
python scripts/p07_sentinel1_clean_plot_v3.py
```

## Landsat p06c cannot find cleaned files

Run p06b before p06c:

```bash
python scripts/p06b_landsat_remove_outliers_per_polygon.py
python scripts/p06c_landsat_plot_ndvi_per_polygon_yearwise.py
```

## Windows path problems

Use forward slashes in YAML:

```yaml
field_shapefile: "C:/Users/Name/Documents/field_shapefile.shp"
```
