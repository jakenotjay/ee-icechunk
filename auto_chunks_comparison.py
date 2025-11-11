"""Pulls the embeddings data and stores it, uses a local grid, takes a simple boolean argument to determine whether to use the recommended io_chunks or the auto chunks method, outputs a time in seconds to write the data to icechunk."""

import argparse
import time

import dask
import ee
import icechunk as ic
import shapely
import xarray as xr
from dask.distributed import Client, LocalCluster
from icechunk.xarray import to_icechunk
from shapely.geometry import box

import ee_ic


def get_embeddings(
    bounds: ee.geometry.Geometry,
    start_date: ee.ee_date.Date,
    end_date: ee.ee_date.Date,
) -> ee.imagecollection.ImageCollection:
    """Returns the embeddings data for a given bounds and date range."""
    embeddings = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterBounds(bounds)
        .filterDate(start_date, end_date)
    )
    return embeddings


def write_region_to_icechunk(
    *,
    session: ic.ForkSession | ic.Session,
    region: dict[str, slice],
    ds: xr.Dataset,
    project: str,
) -> ic.ForkSession | ic.Session:
    """Takes the slice from the dataset and writes to the icechunk store."""

    print(f"Writing region {region} to icechunk")
    ee.Initialize(
        project=project, opt_url="https://earthengine-highvolume.googleapis.com"
    )
    ds = ds.isel(**region)

    to_icechunk(ds.drop_vars("spatial_ref", errors="ignore"), session, region=region)
    print(f"Wrote region {region} to icechunk")
    return session


def create_grid_and_dataset(
    intended_bounds: shapely.geometry.Polygon,
    chunk_size: tuple[int, int],
    region_size: tuple[int, int],
    time_region_size: int,
    io_chunks: dict[str, int] | None = None,
) -> tuple[ee_ic.ChunkGrid, xr.Dataset]:
    """Creates the chunk grid and loads the dataset."""
    start_date = ee.Date("2017-01-01")
    end_date = ee.Date("2024-12-31")

    target_crs = "EPSG:4326"
    target_res = 10

    grid_xmin, grid_ymin, grid_xmax, grid_ymax = intended_bounds.bounds

    grid = ee_ic.ChunkGrid(
        grid_xmin,
        grid_xmax,
        grid_ymin,
        grid_ymax,
        res=target_res,
        crs=target_crs,
        chunk_size=chunk_size,
        region_size=region_size,
        time_region_size=time_region_size,
    )

    ee_bounds = grid.get_ee_bounds()
    ee_proj = grid.get_ee_projection()

    collection = get_embeddings(ee_bounds, start_date, end_date)

    ds = xr.open_dataset(
        collection,
        engine="ee",
        projection=ee_proj,
        geometry=ee_bounds,
        io_chunks=io_chunks,
    )

    grid = grid.configure_from_dataset(ds)

    return grid, ds


def setup_repository(bucket: str, prefix: str, grid: ee_ic.ChunkGrid) -> ic.Repository:
    """Setup the icechunk repository, creating it if it doesn't exist."""
    storage = ic.gcs_storage(bucket=bucket, prefix=prefix, from_env=True)

    try:
        repo = ic.Repository.open(storage)
        print("Opened existing repository")
    except Exception:
        print("Creating new repository")
        template, encoding = grid.get_template_and_encoding()

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull embeddings data and store it using a local grid"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="GCS bucket name (default: dev-epoch-chunks)",
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Google Earth Engine project ID (default: epoch-geospatial-dev)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Prefix for the icechunk store (default: embeddings_local)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        required=True,
        help="Chunk size for the icechunk store (default: 512)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        required=True,
        help="Number of workers to use (default: 4)",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        required=True,
        help="Number of threads per worker (default: 1)",
    )
    args = parser.parse_args()

    bucket = args.bucket
    project = args.project
    prefix = args.prefix
    chunk_size = (args.chunk_size, args.chunk_size)
    n_workers = args.n_workers
    threads_per_worker = args.threads_per_worker

    ee.Authenticate()
    ee.Initialize(
        project=project, opt_url="https://earthengine-highvolume.googleapis.com"
    )

    xmin, ymin, xmax, ymax = (
        35.4607002239093,
        -16.12147572214461,
        35.82928740166918,
        -15.836284989543742,
    )
    intended_bounds = box(xmin, ymin, xmax, ymax)

    region_size = (512, 512)
    time_region_size = 8
    recommended_io_chunks = None

    grid, ds = create_grid_and_dataset(
        intended_bounds,
        chunk_size,
        region_size,
        time_region_size,
        recommended_io_chunks,
    )

    repo = setup_repository(bucket, prefix, grid)
    regions = grid.get_all_regions()

    with LocalCluster(
        n_workers=n_workers, threads_per_worker=threads_per_worker
    ) as cluster:
        with Client(cluster) as client:
            session = repo.writable_session("main")
            tasks = []

            for region in regions:
                fork = session.fork()

                tasks.append(
                    dask.delayed(write_region_to_icechunk)(
                        session=fork, region=region, ds=ds, project=project
                    )
                )

            remote_session = dask.compute(*tasks, scheduler=client)

            session.merge(*remote_session)
            session.commit("Wrote all regions")
            print("Committed all regions")


if __name__ == "__main__":
    import time

    tik = time.time()
    main()
    tok = time.time()
    print(f"Time taken: {tok - tik} seconds")
