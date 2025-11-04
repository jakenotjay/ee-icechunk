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
