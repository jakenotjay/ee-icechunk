from typing import Any

import icechunk as ic
import numpy as np
import xarray as xr
from odc.geo.xr import spatial_dims
from xee.ext import REQUEST_BYTE_LIMIT, EarthEngineStore

TIME_DIMS = ["time", "t"]


def temporal_dim(xx: xr.DataArray | xr.Dataset, relaxed: bool = False) -> str | None:
    """Find the temporal dimension of an xarray object ``xx``

    Checks for the presence of dimensions named:
    ``time, t``

    If ``relaxed=True`` and none of the above dimension names are found,
    assume that the last dimension is the temporal dimension
    (if it doesn't share a name with a spatial dimension)

    Returns:
        str | None: The name of the temporal dimension, or None if no temporal dimension is found
    """

    _dims = [str(dim) for dim in xx.dims]
    dims = set[str](_dims)
    for guess in TIME_DIMS:
        if dims.issuperset({guess}):
            return guess

    if not relaxed:
        return None

    if len(_dims) < 3:
        return None

    s_dims = spatial_dims(xx, relaxed=True)

    last_dim = _dims[-1]

    if s_dims is None:
        return last_dim

    if last_dim in s_dims:
        return None

    return last_dim


def extract_time_config(
    ds: xr.Dataset, relaxed: bool = False
) -> tuple[str | None, Any | None]:
    """Extract time dimension name and coordinates from a dataset.

    Args:
        ds: The xarray dataset to extract time information from
        relaxed: If True, use relaxed temporal dimension detection

    Returns:
        tuple[str | None, Any | None]: (time_dim_name, time_coordinates)
            Returns (None, None) if no temporal dimension is found
    """
    time_dim = temporal_dim(ds, relaxed=relaxed)

    if time_dim is None:
        return None, None

    time_coords = ds[time_dim].data

    return time_dim, time_coords


def extract_band_config(ds: xr.Dataset) -> dict[str, np.dtype]:
    """Extract band names and their dtypes from a dataset.

    Args:
        ds: The xarray dataset to extract band information from

    Returns:
        dict[str, np.dtype]: Mapping of band names to their dtypes
    """
    return {str(band): ds[band].dtype for band in ds.data_vars}


def extract_dataset_config(ds: xr.Dataset, relaxed: bool = False) -> dict[str, Any]:
    """Extract all configuration needed for ChunkGrid from a dataset.

    This is a convenience function that extracts temporal and band information
    from a dataset that can be used to configure a ChunkGrid instance.

    Args:
        ds: The xarray dataset to extract configuration from
        relaxed: If True, use relaxed temporal dimension detection

    Returns:
        dict with keys:
            - time_dim: str | None - name of time dimension
            - time_coords: array-like | None - time coordinate values
            - bands: dict[str, dtype] - mapping of band names to dtypes
            - max_dtype_bytes: int - the size of the largest dtype in the dataset in bytes e.g. np.float32 is 4 bytes
    """
    time_dim, time_coords = extract_time_config(ds, relaxed=relaxed)
    bands = extract_band_config(ds)

    max_dtype_bytes = max([np.dtype(val).itemsize for val in bands.values()])

    return {
        "time_dim": time_dim,
        "time_coords": time_coords,
        "bands": bands,
        "max_dtype_bytes": max_dtype_bytes,
    }


def recommend_io_chunks(
    max_dtype_bytes: int,
    n_time_steps: int,
    min_width: int = 512,
    min_height: int = 512,
) -> dict[str, int]:
    """Earth Engine has a maximum request size of 48MB, we can use this to
    recommend a chunk size, when making requests to the earth engine backend.

    Earth Engine's requesting backend splits requests into tiles, tiles are PER band and they convert image collections into individual images, with bands
    for the time dimension.

    Therefore we prefer to group time steps together.

    Inspired by _auto_chunks from xee.

    In the case that the budget is too small to fit the entire temporal dimension, we use the min_width and min_height to allow for smaller requests.
    """
    log_budget = np.log2(REQUEST_BYTE_LIMIT)

    log_dtype = np.log2(max_dtype_bytes + 1)  # + 1 because ee ships a mask byte

    budget = log_budget - log_dtype

    log_time = np.log2(n_time_steps)

    budget_without_time = budget - log_time

    log_min_width = np.log2(min_width)
    log_min_height = np.log2(min_height)

    if budget_without_time - log_min_width - log_min_height < 0:
        width, height = min_width, min_height
        wd, ht = np.log2(width), np.log2(height)
        time_budget = budget - wd - ht
        index = int(np.rint(2 ** np.floor(time_budget)))

        return {"index": index, "width": width, "height": height}

    # automatically maximises the remaining budget for width and height by rounding to powers of two (even case will result in square, odd case will result in a wide rectangle)
    remainder = np.floor(budget_without_time) / 2
    wd, ht = np.ceil(remainder), np.floor(remainder)

    index = int(n_time_steps)
    width = int(np.rint(2**wd))
    height = int(np.rint(2**ht))

    return {"index": index, "width": width, "height": height}


def compute_optimium_threads_per_worker(
    n_workers: int,
    max_dtype_bytes: int,
    max_concurrent_requests: int,
    n_bands: int,
    region_width: int,
    region_height: int,
    region_depth: int = 1,
) -> int:
    """Computes the optimal number of concurrent threads to run per worker,
    based on the maximum number of concurrent requests that earth engine allows.

    Args:
        n_workers: The number of workers running in parallel
        max_dtype_bytes: The size of the largest dtype in the dataset in bytes e.g. np.float32 is 4 bytes
        max_concurrent_requests: Max number of concurrent requests to EE, defined by your plan
        number_of_regions: The number of regions to process
        n_bands: The number of bands in the dataset
        region_width: the width of the region in pixels
        region_height: the height of the region in pixels
        region_depth: the depth of the region in pixels (i.e. number of time steps at once)

    Returns:
        int: The optimal number of concurrent threads to run per worker
    """

    auto_chunks = EarthEngineStore._auto_chunks
    preferred_chunks = auto_chunks(max_dtype_bytes, REQUEST_BYTE_LIMIT)
    print(f"Preferred chunks: {preferred_chunks}")

    # now we assume that preferred_chunks is essentially the size of a request that XEE will make
    # we can define a density metric by multiplying height, width and index
    # request density is the number of pixels in a request before multiplying by the number of bands
    # and number of bits per pixel
    request_density = (
        preferred_chunks["height"]
        * preferred_chunks["width"]
        * preferred_chunks["index"]
    )

    print(f"Request density: {request_density}")

    region_density = region_width * region_height * region_depth * n_bands

    print(f"Region density: {region_density}")

    max_density = max_concurrent_requests * request_density

    print(f"Max density: {max_density}")

    optimal_threads = max_density // region_density

    print(f"Optimal threads: {optimal_threads}")
    optimal_threads = min(optimal_threads, max_concurrent_requests)

    threads_per_worker = optimal_threads // n_workers

    return threads_per_worker


async def get_written_chunks_set(
    session: ic.Session, band_name: str
) -> set[tuple[int, ...]]:
    """Fetches all written chunks for a given band name (array path) from the icechunk store.

    As per icechunk developers, we should find this performant up to a million chunks being written.
    https://earthmover-community.slack.com/archives/C07NQCBSTB7/p1759151210655289

    Once we achieve performance issues, we can write a cache of the written chunks to a database.
    """
    chunk_set: set[tuple[int, ...]] = set()
    written_chunks = session.chunk_coordinates(f"/{band_name}")

    async for chunk in written_chunks:
        chunk_set.add(chunk)

    return chunk_set
