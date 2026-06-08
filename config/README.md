# Configuration files

This repository is designed so users do **not** edit Python scripts for each new field.
Instead, create a field-specific YAML file:

```bash
cp config/config.example.yaml config/config.yaml
```

Then edit `config/config.yaml`, or generate it automatically:

```bash
python scripts/prepare_config.py \
  --field-name My_Field \
  --field-shapefile "/path/to/field_shapefile.shp" \
  --output-dir "/path/to/Satellite_Data" \
  --ee-project "your-earth-engine-project-id" \
  --write config/config.yaml
```

For crop-mask workflows, use `config.with_crop_mask.example.yaml` or pass
`--crop-mask-shapefile` to `scripts/prepare_config.py`.

`config/config.yaml` is ignored by Git so local paths are not accidentally committed.
