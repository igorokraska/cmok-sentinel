import os
import sys
import warnings
import requests
import numpy as np
import pandas as pd
import rasterio
import pyproj
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import geopandas as gpd

from pathlib import Path
from shapely import wkt
from shapely.ops import transform
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from urllib3.exceptions import InsecureRequestWarning


warnings.simplefilter("ignore", InsecureRequestWarning)

CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
CLOUD_CLASSES = [3, 8, 9, 10]  # SCL cloud flags


def choose_area(directory):
    shp_dir = os.listdir(directory)
    shp_dir = [file for file in shp_dir if file.endswith(".shp")]
    if not shp_dir:
        raise ValueError("No shp files found in this directory")
    for n, file in enumerate(shp_dir):
        sys.stdout.write(f"{n}: {file} \n")
    bound_min, bound_max = 0, len(shp_dir)
    while True:
        choosen_produt = int(input(f"Choose a shp file: "))
        if bound_min <= choosen_produt <= bound_max:
            break
    file_choosen = shp_dir[choosen_produt]
    gdf = gpd.read_file(os.path.join(directory, file_choosen))
    poly = gdf.geometry.iloc[0]
    wkt_string = poly.wkt
    return wkt_string


class SentinelSession:
    def __init__(self, username, password):
        self.session = self._authenticate(username, password)

    def _authenticate(self, username, password):
        url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        r = requests.post(
            url,
            data={
                "client_id": "cdse-public",
                "grant_type": "password",
                "username": username,
                "password": password,
            },
        )
        token = r.json()["access_token"]
        s = requests.Session()
        s.headers["Authorization"] = f"Bearer {token}"
        return s


class SentinelCatalogue:
    def search(self, start, end, cloud, aoi_wkt):
        query = (
            f"{CATALOGUE_URL}/Products?"
            f"$filter=Collection/Name eq 'SENTINEL-2' "
            f"and Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' "
            f"and a/OData.CSC.StringAttribute/Value eq 'S2MSI2A') "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}') "
            f"and Attributes/OData.CSC.DoubleAttribute/any(a:a/Name eq 'cloudCover' "
            f"and a/OData.CSC.DoubleAttribute/Value le {cloud}) "
            f"and ContentDate/Start gt {start} and ContentDate/Start lt {end}"
        )
        products = requests.get(query).json()["value"]
        df = pd.DataFrame(products)
        df = df.assign(**pd.DataFrame.from_records(df.ContentDate.tolist()))
        df["Start"] = pd.to_datetime(df["Start"])
        df.sort_values("Start", inplace=True, ignore_index=True)
        for index, val in enumerate(df["ContentDate"]):
            sys.stdout.write(f"{index}: {val["Start"]} \n")
        bound_min, bound_max = min(df.index), max(df.index)
        while True:
            choosen_produt = input(f"Choose product between {bound_min} and {bound_max} or download all (a): ")
            try:
                choosen_produt = int(choosen_produt)
                if bound_min <= choosen_produt <= bound_max:
                    df_choosen = df.iloc[choosen_produt]
                    df = pd.DataFrame({"Id": [df_choosen["Id"]],
                                       "Name":  [df_choosen["Name"]],
                                       "Start": [df_choosen["ContentDate"]["Start"]]})
                    return df
            except:
                if (choosen_produt == "a"):
                    df = pd.DataFrame({"Id": df["Id"].to_list(),
                                       "Name":  df["Name"].to_list(),
                                       "Start": df["ContentDate"].apply(pd.Series)["Start"].to_list()})
                    return df


class SentinelProduct:
    def __init__(self, session, product_id, product_name, product_date):
        self.session = session
        self.id = product_id
        self.name = product_name
        self.date = str(product_date).replace(":", "_")
        self.root = self._download_metadata()

    def _download_metadata(self):
        url = f"{CATALOGUE_URL}/Products({self.id})/Nodes({self.name})/Nodes(MTD_MSIL2A.xml)/$value"
        response = self.session.get(url, allow_redirects=False)
        while response.status_code in (301, 302, 303, 307):
            url = response.headers["Location"]
            response = self.session.get(url, allow_redirects=False)
        xml_r = self.session.get(url, allow_redirects=True, verify=False)
        xml_path = Path("MTD_MSIL2A.xml")
        xml_path.write_bytes(xml_r.content)
        return ET.parse(xml_path).getroot()

    def download_band(self, band_key):
        for el in self.root.iter("IMAGE_FILE"):
            if band_key in el.text:
                rel = el.text + ".jp2"
                break
        else:
            raise RuntimeError(f"Band {band_key} not found")

        url = f"{CATALOGUE_URL}/Products({self.id})/Nodes({self.name})"
        for p in rel.split("/"):
            url += f"/Nodes({p})"
        url += "/$value"
        r = self.session.get(url, allow_redirects=True, verify=False)
        out_dir = Path("data") / self.date.replace(" ", "").replace("+", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / Path(rel).name
        out.write_bytes(r.content)
        return out


class AOIProcessor:
    @staticmethod
    def crop(array, meta, aoi_wkt):
        geom = wkt.loads(aoi_wkt)
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", meta["crs"], always_xy=True
        ).transform
        geom = transform(transformer, geom)

        minx, miny, maxx, maxy = geom.bounds
        r1, c1 = rowcol(meta["transform"], minx, maxy)
        r2, c2 = rowcol(meta["transform"], maxx, miny)

        r1, r2 = sorted([r1, r2])
        c1, c2 = sorted([c1, c2])

        cropped = array[r1:r2, c1:c2]
        meta = meta.copy()
        meta.update(
            height=cropped.shape[0],
            width=cropped.shape[1],
            transform=rasterio.transform.from_origin(
                meta["transform"][2] + c1 * meta["transform"][0],
                meta["transform"][5] + r1 * meta["transform"][4],
                meta["transform"][0],
                meta["transform"][4],
            ),
        )
        return cropped, meta


class IndexDefinition:
    DEFINITIONS = {
        "NDVI": {
            "bands": {
                "numerator": "B08",
                "denominator": "B04",
                "SCL": "SCL",
            },
            "resolutions": {
                "B08": "R10",
                "B04": "R10",
                "SCL": "R20",
            },
            "formula": lambda a, b: (a - b) / (a + b),
            "cmap": "RdYlGn",
            "folder": "ndvi",
                },
        "NDMI": {
            "bands": {
                "numerator": "B08",   # NIR
                "denominator": "B11", # SWIR
                "SCL": "SCL",
            },
            "resolutions": {
                "B08": "R10",
                "B11": "R20",
                "SCL": "R20",
            },
            "formula": lambda a, b: (a - b) / (a + b),
            "cmap": "RdYlBu",
            "folder": "ndmi",
                },
        "NDRE": {
            "bands": {
                "numerator": "B08",   # NIR
                "denominator": "B05", # Red Edge
                "SCL": "SCL",
            },
            "resolutions": {
                "B08": "R10",
                "B05": "R20",
                "SCL": "R20",
            },
            "formula": lambda a, b: (a - b) / (a + b),
            "cmap": "RdYlGn",
            "folder": "ndre",
                },
        "NDWI": {
            "bands": {
                "numerator": "B03",   # Green
                "denominator": "B08", # NIR
                "SCL": "SCL",
            },
            "resolutions": {
                "B03": "R10",
                "B08": "R10",
                "SCL": "R20",
            },
            "formula": lambda a, b: (a - b) / (a + b),
            "cmap": "GnBu",
            "folder": "ndwi",
        },
    }

    def __init__(self, name):
        self.name = name
        self.cfg = self.DEFINITIONS[name]


class BandSet:
    def __init__(self, product, index_def):
        self.product = product
        self.index_def = index_def
        self.downloaded = {}
        self.data_dir = Path(f"data_{index_def.cfg['folder']}") / product.date
        self.tiff_dir = Path(f"tiffs_{index_def.cfg['folder']}")

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tiff_dir.mkdir(exist_ok=True)

        self.tiff_path = self.tiff_dir / f"{product.date}.tif"

    def _download_raw_band(self, rel_path):
        url = f"{CATALOGUE_URL}/Products({self.product.id})/Nodes({self.product.name})"
        for p in rel_path.split("/"):
            url += f"/Nodes({p})"
        url += "/$value"
        r = self.product.session.get(url, allow_redirects=False)
        while r.status_code in (301, 302, 303, 307):
            url = r.headers["Location"]
            r = self.product.session.get(url, allow_redirects=False)
        r = self.product.session.get(url, verify=False, allow_redirects=True)
        if r.status_code != 200:
            raise RuntimeError(f"Failed download: HTTP {r.status_code}")

        if r.headers.get("Content-Type", "").startswith("text/html"):
            raise RuntimeError(
                f"Downloaded HTML instead of JP2 for {rel_path}"
            )
        out = self.data_dir / Path(rel_path).name
        out.write_bytes(r.content)
        if out.stat().st_size < 100_000:
            raise RuntimeError(f"Suspiciously small JP2: {out}")
        return str(out)

    def download(self):
        needed = self.index_def.cfg["resolutions"]
        # find JP2 paths in metadata
        for el in self.product.root.iter("IMAGE_FILE"):
            for band, res in needed.items():
                if band in el.text and res in el.text:
                    self.downloaded[band] = el.text + ".jp2"
        if set(self.downloaded) != set(needed):
            raise RuntimeError(
                f"Missing bands for {self.index_def.name}: "
                f"{set(needed) - set(self.downloaded)}"
            )

        # download JP2 files
        for band, rel_path in self.downloaded.items():
            path = self._download_raw_band(rel_path)
            self.downloaded[band] = path
            sys.stdout.write(f"INFO Downloaded {band} -> {path} \n")

        self.downloaded["tiff_path"] = str(self.tiff_path)
        return self.downloaded


class SpectralIndex:
    DEFINITIONS = {
        "NDVI": ("B08", "B04", "RdYlGn"),
        "NDMI": ("B08", "B11", "RdYlBu"),
        "NDRE": ("B08", "B05", "RdYlGn"),
        "NDWI": ("B03", "B08", "Blues"),
    }

    def __init__(self, name):
        self.name = name
        self.num_band, self.den_band, self.cmap = self.DEFINITIONS[name]

    def compute(self, a, b, scl):
        mask = np.isin(scl, CLOUD_CLASSES)
        a = a.filled(np.nan)
        b = b.filled(np.nan)
        den = a + b
        idx = np.full(a.shape, np.nan, "float32")
        valid = (~mask) & (den != 0)
        idx[valid] = (a[valid] - b[valid]) / den[valid]
        return idx


class SentinelWorkflow:
    def __init__(self, username, password):
        self.session = SentinelSession(username, password).session
        self.catalogue = SentinelCatalogue()

    @staticmethod
    def _read(path, ref_shape=None):
        with rasterio.open(path) as src:
            if ref_shape:
                arr = src.read(
                    1, out_shape=ref_shape, resampling=Resampling.bilinear
                )
            else:
                arr = src.read(1)
            return np.ma.masked_invalid(arr.astype("float32")), src.meta.copy()


def process_index(bandset, index_def, aoi, save, show):
    def read(path, shape=None, resampling=Resampling.nearest):
        with rasterio.open(path) as src:
            if shape:
                arr = src.read(1, out_shape=shape, resampling=resampling)
            else:
                arr = src.read(1)
            return np.ma.masked_invalid(arr.astype("float32")), src.meta.copy()

    band_names = list(index_def.cfg["bands"].keys())
    band_names.remove("SCL")
    band_a = index_def.cfg["bands"]["numerator"]
    band_b = index_def.cfg["bands"]["denominator"]

    arr_a, meta = read(bandset[band_a])
    arr_b, _ = read(
        bandset[band_b],
        shape=arr_a.shape,
        resampling=Resampling.bilinear,
    )

    scl, _ = read(
        bandset["SCL"],
        shape=arr_a.shape,
        resampling=Resampling.nearest,
    )

    mask = np.isin(scl, CLOUD_CLASSES)
    mask |= arr_a.mask | arr_b.mask

    a = arr_a.filled(np.nan)
    b = arr_b.filled(np.nan)

    den = a + b
    out = np.full(arr_a.shape, np.nan, dtype="float32")

    valid = (~mask) & (den != 0)
    out[valid] = index_def.cfg["formula"](a[valid], b[valid])
    out, meta = AOIProcessor.crop(out, meta, aoi)

    meta.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=np.nan,
    )

    if save:
        with rasterio.open(bandset["tiff_path"], "w", **meta) as dst:
            dst.write(out, 1)


    if show:
        plt.imshow(out, cmap=index_def.cfg["cmap"], vmin=-1, vmax=1)
        plt.colorbar()
        plt.show()

    return out


def run_sentinel_index_workflow(
    username,
    password,
    index_name,
    shp_dir,
    start,
    end,
    cloud=20,
    save=True,
    show=True,
):
    """
    Master workflow for downloading Sentinel-2 L2A data and computing spectral indices.

    Parameters
    ----------
    username : str
        Copernicus Data Space username
    password : str
        Copernicus Data Space password
    index_name : str
        Index name (NDVI, NDWI, NDMI, NDRE)
    shp_dir : str
        Directory containing AOI shapefile
    start : str
        Start date (ISO format)
    end : str
        End date (ISO format)
    cloud : int, optional
        Maximum cloud cover percentage
    save : bool, optional
        Save GeoTIFF output
    show : bool, optional
        Show index plot
    """
    available_ind = ["NDVI", "NDMI", "NDRE", "NDWI"]
    if index_name not in available_ind:
        raise ValueError("Available indices are: NDVI, NDMI, NDRE or NDWI")
    try:
        start_date = pd.to_datetime(start, format="%Y-%m-%d")
        end_date = pd.to_datetime(end, format="%Y-%m-%d")
    except:
        raise ValueError("start and end should be dates in format yyyy-mm-dd")
    if type(cloud) != int or cloud < 0 or cloud > 100:
        raise ValueError("cloud should be integer in range 0-100")
    aoi = choose_area(shp_dir)

    wf = SentinelWorkflow(username, password)
    df = wf.catalogue.search(
        start=start,
        end=end,
        cloud=cloud,
        aoi_wkt=aoi,
    )

    index_def = IndexDefinition(index_name)

    for _, row in df.iterrows():
        product = SentinelProduct(
            wf.session,
            row["Id"],
            row["Name"],
            row["Start"],
        )

        bandset = BandSet(product, index_def).download()

        process_index(
            bandset=bandset,
            index_def=index_def,
            aoi=aoi,
            save=save,
            show=show,
        )
