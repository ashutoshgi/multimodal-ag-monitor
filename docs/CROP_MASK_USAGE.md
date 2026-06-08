# Crop mask usage

Some field boundaries include non-crop areas such as roads, ditches, farmstead areas, grass strips, or bare soil. If a crop mask shapefile is available, the workflow can intersect the field boundary with the crop mask before data download and time-series extraction.

## Create a masked config

```bash
python scripts/prepare_config.py \
  --field-name My_Field_Masked \
  --field-shapefile "/absolute/path/to/field_shapefile.shp" \
  --crop-mask-shapefile "/absolute/path/to/crop_mask.shp" \
  --output-dir "/absolute/path/to/Satellite_Data" \
  --ee-project "your-earth-engine-project-id" \
  --write config/config.yaml
```

The resulting config contains:

```yaml
mask:
  apply_crop_mask: true
  crop_mask_shapefile: "/absolute/path/to/crop_mask.shp"
```

When enabled, p01 uses:

```text
field boundary ∩ crop mask
```

for downloads and field-level summaries.

## Check the mask result

After p01 runs, inspect the QA/QC outputs in:

```text
Satellite_Data/QA_QC/
```

and the cleaned shapefile saved in the output directory.
