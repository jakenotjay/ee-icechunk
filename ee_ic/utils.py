from typing import Any

import numpy as np
import xarray as xr
from odc.geo.xr import spatial_dims

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
        if dims.issuperset(guess):
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
    """
    time_dim, time_coords = extract_time_config(ds, relaxed=relaxed)
    bands = extract_band_config(ds)

    return {
        "time_dim": time_dim,
        "time_coords": time_coords,
        "bands": bands,
    }
