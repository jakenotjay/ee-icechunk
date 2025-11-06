"""Pulls the embeddings data and stores it, uses a global grid but locally defined AOI."""

import argparse

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
    local_region: dict[str, slice],
    ds: xr.Dataset,
    project: str,
) -> ic.ForkSession | ic.Session:
    """Writes the region to the icechunk store, using the local region to define the slices taken from the dataset."""
    ee.Initialize(
        project=project, opt_url="https://earthengine-highvolume.googleapis.com"
    )
    print(f"Writing region {region} to icechunk")
    ds = ds.isel(**local_region)
    to_icechunk(ds.drop_vars("spatial_ref", errors="ignore"), session, region=region)
    print(f"Wrote region {region} to icechunk")
    return session


def create_grid_and_dataset(
    intended_bounds: shapely.geometry.Polygon,
    chunk_size: tuple[int, int],
    region_size: tuple[int, int],
) -> tuple[ee_ic.ChunkGrid, xr.Dataset]:
    """Creates the chunk grid and loads the dataset."""
    start_date = ee.Date("2020-01-01")
    end_date = ee.Date("2024-12-31")

    target_crs = "EPSG:4326"
    target_res = 10

    grid_xmin, grid_ymin, grid_xmax, grid_ymax = (
        -180,
        -90,
        180,
        90,
    )

    grid = ee_ic.ChunkGrid(
        grid_xmin,
        grid_xmax,
        grid_ymin,
        grid_ymax,
        res=target_res,
        crs=target_crs,
        chunk_size=chunk_size,
        region_size=region_size,
    )

    grid_aligned_ee_bounds = grid.get_grid_aligned_ee_bounds(intended_bounds)
    ee_proj = grid.get_ee_projection()

    collection = get_embeddings(grid_aligned_ee_bounds, start_date, end_date)

    ds = xr.open_dataset(
        collection, engine="ee", projection=ee_proj, geometry=grid_aligned_ee_bounds
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


def verify_region_shapes(
    regions: list[dict[str, slice]],
    local_regions: list[dict[str, slice]],
    grid: ee_ic.ChunkGrid,
) -> None:
    """Verify that each region and local_region pair has the same shape.

    Args:
        regions: Global regions from grid.get_regions_from_bounds()
        local_regions: Local regions from grid.get_subregions_in_bounds()
        grid: The ChunkGrid instance for dimension names

    Raises:
        ValueError: If regions and local_regions have different lengths or shapes don't match
    """
    print(f"Verifying {len(regions)} regions and {len(local_regions)} local_regions")
    if len(regions) != len(local_regions):
        raise ValueError(
            f"Mismatch: {len(regions)} regions but {len(local_regions)} local_regions"
        )

    for i, (region, local_region) in enumerate(
        zip(regions, local_regions, strict=True)
    ):
        region_shape = (
            region[grid.y_dim].stop - region[grid.y_dim].start,
            region[grid.x_dim].stop - region[grid.x_dim].start,
        )
        local_shape = (
            local_region[grid.y_dim].stop - local_region[grid.y_dim].start,
            local_region[grid.x_dim].stop - local_region[grid.x_dim].start,
        )

        if region_shape != local_shape:
            raise ValueError(
                f"Shape mismatch at pair {i}: region shape {region_shape} "
                f"!= local_region shape {local_shape}"
            )

    print(f"✓ All {len(regions)} region pairs have matching shapes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull embeddings data and store it using a global grid with a locally defined AOI"
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
        help="Prefix for the icechunk store (default: embeddings_global)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        required=True,
        help="Chunk size for the icechunk store (default: 512)",
    )
    parser.add_argument(
        "--region-size",
        type=int,
        required=True,
        help="Region size for the icechunk store (default: 1024)",
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
        help="Number of threads per worker (default: 4)",
    )
    args = parser.parse_args()

    bucket = args.bucket
    project = args.project
    prefix = args.prefix

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

    chunk_size = (args.chunk_size, args.chunk_size)
    region_size = (args.region_size, args.region_size)

    grid, ds = create_grid_and_dataset(intended_bounds, chunk_size, region_size)

    print(f"Dataset: {ds}")

    print(f"Got dataset with lon and lat dimension lengths of {ds.sizes}")
    print(
        f"Got grid with width and height {grid.geobox.width} and {grid.geobox.height}"
    )
    repo = setup_repository(bucket, prefix, grid)

    regions = grid.get_regions_from_bounds(intended_bounds)
    local_regions = grid.get_subregions_in_bounds(intended_bounds)

    verify_region_shapes(regions, local_regions, grid)

    # threads_per_worker = ee_ic.utils.compute_optimium_threads_per_worker(
    #     n_workers=4,
    #     max_dtype_bytes=8,
    #     max_concurrent_requests=500,
    #     n_bands=64,
    #     region_width=1024,
    #     region_height=1024,
    #     region_depth=5,
    # )

    # print(f"Threads per worker: {threads_per_worker}")

    with LocalCluster(
        n_workers=args.n_workers, threads_per_worker=args.threads_per_worker
    ) as cluster:
        with Client(cluster) as client:
            session = repo.writable_session("main")

            tasks = []

            for region, local_region in zip(regions, local_regions, strict=True):
                fork = session.fork()

                tasks.append(
                    dask.delayed(write_region_to_icechunk)(
                        session=fork,
                        region=region,
                        local_region=local_region,
                        ds=ds,
                        project=project,
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
