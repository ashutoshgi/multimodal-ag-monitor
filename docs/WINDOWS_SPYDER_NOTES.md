# Windows and Spyder notes

## Recommended setup

Use Conda/Mamba because geospatial packages are easier to install from conda-forge.

```bash
conda env create -f environment.yml
conda activate multimodal-eo
earthengine authenticate
```

Then open Spyder from the same environment, or install Spyder in that environment if needed.

## Running in Spyder

1. Set the working directory to the repository root folder.
2. Create `config/config.yaml`.
3. Open a script from the `scripts/` folder.
4. Run scripts one at a time.

Do not set the working directory to the `scripts/` folder unless you also set `EO_CONFIG` manually.

## Windows paths in YAML

Prefer forward slashes:

```yaml
field_shapefile: "C:/Users/Name/Documents/My_Field/field_shapefile.shp"
output_dir: "C:/Users/Name/Documents/My_Field/Satellite_Data"
```
