# Installation

## Recommended: Conda/Mamba

Geospatial Python packages depend on GDAL. Conda/Mamba is usually the most reliable installation method on Linux, Windows, and macOS.

```bash
conda env create -f environment.yml
conda activate multimodal-eo
earthengine authenticate
```

## Pip option

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
earthengine authenticate
```

## Earth Engine setup

Users need a Google Earth Engine account and a valid project ID if required by their account.

```bash
earthengine authenticate
```

Then place the Earth Engine project ID in `config/config.yaml`:

```yaml
gee:
  project: "your-earth-engine-project-id"
```
