import argparse
from .sentinel_tool import SentinelWorkflow, IndexDefinition, choose_area

def main():
    parser = argparse.ArgumentParser(description="Sentinel-2 spectral index processing")
    parser.add_argument("--index", type=str, choices=["NDVI", "NDMI", "NDRE", "NDWI"], required=True, help="Spectral index to compute")
    parser.add_argument("--start", type=str, required=True, help="Start date, e.g., 2025-01-01")
    parser.add_argument("--end", type=str, required=True, help="End date, e.g., 2025-03-01")
    parser.add_argument("--cloud", type=float, default=20, help="Maximum cloud cover percentage")
    parser.add_argument("--aoi", type=str, required=True, help="Directory with shapefile for AOI")
    parser.add_argument("--save", action="store_true", help="Save the result as GeoTIFF")
    parser.add_argument("--show", action="store_true", help="Show the result as plot")
    parser.add_argument("--user", type=str, required=True, help="Copernicus username")
    parser.add_argument("--password", type=str, required=True, help="Copernicus password")
    args = parser.parse_args()

    # Choose AOI (shapefile) automatically
    aoi_wkt = choose_area(args.aoi)

    # Run workflow
    wf = SentinelWorkflow(args.user, args.password)
    wf.run(
        index_name=args.index,
        start=args.start + "T00:00:00.000Z",
        end=args.end + "T00:00:00.000Z",
        cloud=args.cloud,
        aoi=aoi_wkt,
        save=args.save,
        show=args.show,
    )

if __name__ == "__main__":
    main()
