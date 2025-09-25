"""An example of how to use the ee-icechunk library to write the ERA-5 dataset to an icechunk store."""

import ee
import xarray as xr
import ee_ic
import icechunk as ic
from icechunk.xarray import to_icechunk
from typing import Union
import dask
from dask.distributed import Client
import argparse


def get_sentinel_2_composites(
    bounds: ee.Geometry.Polygon,
    start_date: ee.Date,
    count: int,
    interval: int,
    interval_units: str,
) -> ee.ImageCollection:
    """Returns a composite of sentinel-2 images for a given region, date range, uses cloud score + for masking

    Args:
        bounds: The bounds of the region to get the composites for
        start_date: The start date of the date range
        count: The number of composites to get
        interval: The interval between composites
        interval_units: The units of the interval (year, month, week, day, hour, minute, second)

    Returns:
        ee.ImageCollection: The sentinel-2 composites
    """

    start_date: ee.Date = ee.Date(start_date)
    count: ee.Number = ee.Number(count)
    interval: ee.Number = ee.Number(interval)
    interval_units: ee.String = ee.String(interval_units)

    QA_BAND = "cs_cdf"
    CLEAR_THRESHOLD = 0.60

    BAND_NAMES = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B11",
        "B12",
        "AOT",
        "WVP",
    ]

    # remove the original qa bands from sentinel-2
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").select(BAND_NAMES)
    cs_plus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

    def create_composite(n):
        begin = start_date.advance(interval.multiply(n), interval_units)
        end = begin.advance(interval, interval_units)

        composite = (
            s2.filterBounds(bounds)
            .filterDate(begin, end)
            .linkCollection(cs_plus, [QA_BAND])
            .map(lambda img: img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD)))
            .mean()
            .set({"system:time_start": begin.millis(), "system:time_end": end.millis()})
        )

        return composite

    images = ee.List.sequence(0, count.subtract(1)).map(create_composite)

    # we must ensure that images are consistently named and in a given order by selecting
    # and renaming the bands
    final_band_names = BAND_NAMES + [QA_BAND]

    return ee.ImageCollection.fromImages(images).select(
        final_band_names, final_band_names
    )


def write_region_to_icechunk(
    *,
    session: Union[ic.ForkSession, ic.Session],
    region: dict[str, slice],
    ds: xr.Dataset,
) -> None:
    """Takes the slice from the dataset and writes to the icechunk store."""
    print(f"Writing region {region} to icechunk")
    ds = ds.isel(**region)

    to_icechunk(ds.drop_vars("spatial_ref", errors="ignore"), session, region=region)
    print(f"Wrote region {region} to icechunk")
    return session


def setup_earth_engine(project: str) -> None:
    """Initialize Earth Engine with the given project."""
    ee.Authenticate(auth_mode="gcloud")
    ee.Initialize(
        project=project, opt_url="https://earthengine-highvolume.googleapis.com"
    )


def create_grid_and_dataset() -> tuple[ee_ic.ChunkGrid, xr.Dataset]:
    """Create the chunk grid and load the Sentinel-2 dataset."""
    start_date = ee.Date("2018-01-01")
    count = 8  # july 2025
    interval = 1
    interval_units = "year"

    target_crs = "EPSG:4326"
    target_res = 10

    xmin, ymin, xmax, ymax = (
        35.4607002239093,
        -16.12147572214461,
        35.82928740166918,
        -15.836284989543742,
    )

    grid = ee_ic.ChunkGrid(
        xmin,
        xmax,
        ymin,
        ymax,
        res=target_res,
        crs=target_crs,
        chunk_size=(512, 512),
        region_size=(1024, 1024),
    )

    ee_proj = grid.get_ee_projection()
    print(ee_proj.getInfo())
    ee_bounds = grid.get_ee_bounds()
    print(ee_bounds.getInfo())
    collection = get_sentinel_2_composites(
        ee_bounds, start_date, count, interval, interval_units
    )

    # simple way to check all composites are good
    # print(collection.aggregate_array("system:band_names").getInfo())

    ds = xr.open_dataset(
        collection, engine="ee", projection=ee_proj, geometry=ee_bounds
    )

    print("loaded ds")
    print(ds)

    return grid, ds


def setup_repository(
    bucket: str, prefix: str, grid: ee_ic.ChunkGrid, ds: xr.Dataset
) -> ic.Repository:
    """Setup the icechunk repository, creating it if it doesn't exist."""
    storage = ic.gcs_storage(bucket=bucket, prefix=prefix, from_env=True)

    try:
        repo = ic.Repository.open(storage)
        print("Opened existing repository")
    except Exception:
        print("Creating new repository")
        template, encoding = grid.get_template_and_encoding(ds)

        repo = ic.Repository.create(storage)
        session = repo.writable_session("main")

        template.to_zarr(
            session.store,
            compute=False,
            mode="w",
            encoding=encoding,
            consolidated=False,
        )

        session.commit("Wrote template")

    return repo


def write_data_to_icechunk(
    repo: ic.Repository, grid: ee_ic.ChunkGrid, ds: xr.Dataset
) -> None:
    """Write the dataset to icechunk in parallel chunks."""

    all_regions = grid.get_all_regions(
        ds
    )  # to get as a list of dicts of slices to directly manipulate an array
    # all_regions_ee = grid.get_all_region_ee_bounds() # to get as a feature collection to use in ee operations

    with Client() as client:
        session = repo.writable_session("main")
        tasks = []

        for region in all_regions:
            fork = session.fork()

            tasks.append(
                dask.delayed(write_region_to_icechunk)(
                    session=fork, region=region, ds=ds
                )
            )

        remote_session = dask.compute(*tasks, scheduler=client)

        session.merge(*remote_session)
        session.commit("Wrote all regions")
        print("Committed all regions")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Write Sentinel-2 dataset to an icechunk store"
    )
    parser.add_argument(
        "--project", required=True, help="Google Earth Engine project ID"
    )
    parser.add_argument(
        "--bucket", required=True, help="GCS bucket name for icechunk storage"
    )
    parser.add_argument(
        "--prefix", required=True, help="GCS prefix path for icechunk storage"
    )

    return parser.parse_args()


def main() -> None:
    """Main function that orchestrates the entire process."""
    args = parse_arguments()

    setup_earth_engine(args.project)
    grid, ds = create_grid_and_dataset()
    repo = setup_repository(args.bucket, args.prefix, grid, ds)
    write_data_to_icechunk(repo, grid, ds)

    print("Done")


if __name__ == "__main__":
    main()
