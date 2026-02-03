# Sentinel-2 Spectral Index Tool

This tool downloads Sentinel-2 Level-2A data from the Copernicus Data Space
and computes common spectral indices (NDVI, NDWI, NDMI, NDRE) over a user-defined AOI.

The output is saved as GeoTIFF files and can optionally be visualized.

---

## Features

- Sentinel-2 L2A (BOA) data
- AOI defined by shapefile
- Automatic band downloading
- Cloud masking using SCL
- AOI cropping
- GeoTIFF output
- Indices supported:
  - NDVI
  - NDWI
  - NDMI
  - NDRE

---

## Requirements

- Python 3.9+
- Required libraries:
  - requests
  - numpy
  - pandas
  - rasterio
  - geopandas
  - shapely
  - pyproj
  - matplotlib

---

## Directory Structure

    ├─ project/
    |    ├─ shp/ # AOI shapefiles
    |    ├─ data_<index>/ # Downloaded JP2  bands
    |    ├─ tiffs_<index>/ # Output GeoTIFFs
    |    ├─ sentinel_tool.py
    |    └─ README.md


---


## Installation

```bash
git clone https://github.com/yourusername/sentinel_tool.git
cd sentinel_tool
pip install -e .
```
---

## Usage


First, one needs to create an account in https://dataspace.copernicus.eu/

User name and password will be used in downloading data


### Via Python script

1. Provide directory to your shapefiles
2. Run the script and select the AOI when prompted
3. The script will:
   - search Sentinel-2 products
   - download required bands
   - compute the selected index
   - save results as GeoTIFFs


Example:

```python
run_sentinel_index_workflow(
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    index_name="NDWI",
    shp_dir="./shp",
    start="2025-01-01T00:00:00.000Z",
    end="2025-03-28T00:00:00.000Z",
    cloud=20,
    save=True,
    show=True,
)
```

### Via Command Line Interface (CLI)

After installing, you can use the CLI: 

```bash
sentinel-tool --index NDVI --start 2025-01-01 --end 2025-03-01 --cloud 20 --aoi ./sentinel_tool/shp --user your_username --password your_password --save --show
```

where:

-index: Spectral index to compute (NDVI, NDMI, NDRE, NDWI)

-start / --end: Date range

-cloud: Maximum cloud cover (%)

-aoi: Directory containing shapefile for AOI

-user / --password: Copernicus credentials

-save: Save output as GeoTIFF

-show: Display the result using Matplotlib