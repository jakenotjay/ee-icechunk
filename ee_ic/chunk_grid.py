"""Defines the chunk grid class, a mapping of a region of space to pixel coordinates and the writable chunks that cover it."""

from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_zeros, spatial_dims
from pyproj import CRS, Proj
from shapely.geometry import Polygon, box
import geopandas as gpd
import xarray as xr
import ee


class ChunkGrid:
    """Class defining a range of chunks and regions (groups of chunks) to cover for a given catalog.

    Args:
        xmin (float): The minimum x coordinate of the chunk grid in target crs
        xmax (float): The maximum x coordinate of the chunk grid in target crs
        ymin (float): The minimum y coordinate of the chunk grid in target crs
        ymax (float): The maximum y coordinate of the chunk grid in target crs
        res (float): The resolution of the chunk grid in meters
        crs (str): The target crs
        chunk_size (tuple[int, int]): The size of writable chunks in pixels
        region_size (tuple[int, int]): The size of regions in pixels
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        res: float,
        crs: str = "EPSG:3857",
        chunk_size: tuple[int, int] = (1000, 1000),
        region_size: tuple[int, int] = (4000, 4000),
    ):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.res = res
        self.crs = crs
        self.chunk_size = chunk_size
        self.region_size = region_size
        self._geobox = None

    @property
    def dst_crs(self) -> CRS:
        if isinstance(self.crs, Proj):
            return self.crs.crs

        return CRS.from_string(self.crs)

    def get_geobox(self) -> GeoBox:
        if not self.dst_crs.is_projected:
            # compute the geobox in 3857 first then convert to the target crs
            print(
                "Warning: Target CRS is not projected, so will compute geobox in 3857 then convert to target crs"
            )
            geoseries = gpd.GeoSeries(
                [box(self.xmin, self.ymin, self.xmax, self.ymax)], crs=self.dst_crs
            )
            # convert to 3857
            geoseries = geoseries.to_crs("EPSG:3857")
            xmin, ymin, xmax, ymax = geoseries.total_bounds

            geobox = GeoBox.from_bbox(
                (xmin, ymin, xmax, ymax), crs="EPSG:3857", resolution=self.res
            )
            return geobox.to_crs(self.dst_crs)

        return GeoBox.from_bbox(
            (self.xmin, self.ymin, self.xmax, self.ymax),
            crs=self.dst_crs,
            resolution=self.res,
        )

    @property
    def geobox(self) -> GeoBox:
        if self._geobox is None:
            self._geobox = self.get_geobox()

        return self._geobox

    @property
    def datacube_shape(self) -> tuple[int, int]:
        """The shape of the datacube in (height, width) in pixels."""
        return self.geobox.height, self.geobox.width

    def get_bounds_from_region(self, region: dict[str, slice]) -> Polygon:
        """For a given region (in pixels), returns the bounds in the destination CRS"""

        # check region is within the datacube shape
        if region["x"].start < 0 or region["y"].start < 0:
            raise ValueError("Region is outside the datacube")
        if (
            region["x"].stop > self.datacube_shape[1]
            or region["y"].stop > self.datacube_shape[0]
        ):
            raise ValueError("Region is outside the datacube")

        affine = self.geobox.affine
        x_min, y_max = affine * (region["x"].start, region["y"].start)
        x_max, y_min = affine * (region["x"].stop, region["y"].stop)

        return box(x_min, y_min, x_max, y_max)

    def get_region_from_bounds(self, bounds: Polygon) -> dict[str, slice]:
        """Converts a bounds polygon to a slice of the datacube."""
        xmin, ymin, xmax, ymax = bounds.bounds
        x_start, y_start = ~self.geobox.affine * (xmin, ymax)
        x_end, y_end = ~self.geobox.affine * (xmax, ymin)
        x_start, y_start = int(x_start), int(y_start)
        x_end, y_end = int(x_end), int(y_end)

        # cap the region to the datacube shape
        x_start = max(x_start, 0)
        y_start = max(y_start, 0)
        x_end = min(x_end, self.datacube_shape[1])
        y_end = min(y_end, self.datacube_shape[0])

        # generate the slices and check their size
        if x_end - x_start == 0 or y_end - y_start == 0:
            raise ValueError("Region is too small to be processed")

        return {
            "x": slice(x_start, x_end),
            "y": slice(y_start, y_end),
        }

    def get_all_regions(self) -> list[dict[str, slice]]:
        """Returns a list of regions (in pixels) that cover the datacube."""
        # check x step and y step are integer multiples of the chunk size

        x_step, y_step = self.region_size

        if x_step % self.chunk_size[1] != 0 or y_step % self.chunk_size[0] != 0:
            raise ValueError(
                "x_step and y_step must be integer multiples of the chunk size"
            )

        # create list of reginos using range
        regions = []
        for y in range(0, self.datacube_shape[0], y_step):
            for x in range(0, self.datacube_shape[1], x_step):
                max_x = min(x + x_step, self.datacube_shape[1])
                max_y = min(y + y_step, self.datacube_shape[0])

                regions.append(
                    {
                        "x": slice(x, max_x),
                        "y": slice(y, max_y),
                    }
                )

        return regions

    def get_regions_from_bounds(self, bounds: Polygon) -> list[dict[str, slice]]:
        """Converts region coordinates to pixel bounds, and then iterates over the nearest chunks to generate regions."""
        chunk_step_x, chunk_step_y = self.chunk_size
        region_step_x, region_step_y = self.region_size

        affine = self.geobox.affine

        xmin, ymin, xmax, ymax = bounds.bounds

        # start from top left hence flipped y
        x_start, y_start = ~affine * (xmin, ymax)
        x_end, y_end = ~affine * (xmax, ymin)

        x_start, y_start = int(x_start), int(y_start)
        x_end, y_end = int(x_end), int(y_end)

        # floor to get the minimum chunk bounds
        x_start = x_start // chunk_step_x * chunk_step_x
        y_start = y_start // chunk_step_y * chunk_step_y

        # ceiling to get the maximum chunk bounds
        x_end = (x_end + chunk_step_x) // chunk_step_x * chunk_step_x
        y_end = (y_end + chunk_step_y) // chunk_step_y * chunk_step_y

        regions = []
        for y in range(y_start, y_end, region_step_y):
            for x in range(x_start, x_end, region_step_x):
                max_x = min(x + region_step_x, x_end, self.datacube_shape[1])
                max_y = min(y + region_step_y, y_end, self.datacube_shape[0])

                region = {
                    "x": slice(x, max_x),
                    "y": slice(y, max_y),
                }

                regions.append(region)

        return regions

    def get_ee_projection(self) -> ee.projection.Projection:
        """Returns an earth engine projection object for the chunk grid."""

        crs = self.dst_crs
        geobox = self.geobox

        x_scale, x_shearing, x_translation, y_shearing, y_scale, y_translation = (
            geobox.affine[:6]
        )

        return ee.Projection(
            crs=crs.to_string(),
            transform=[
                x_scale,
                x_shearing,
                x_translation,
                y_shearing,
                y_scale,
                y_translation,
            ],
        )

    def get_ee_bounds(self) -> ee.geometry.Geometry:
        """Returns an earth engine geometry object for the chunk grid, always in EPSG:4326."""
        return ee.Geometry.BBox(self.xmin, self.ymin, self.xmax, self.ymax)

    def get_template_and_encoding(self, ds: xr.Dataset) -> tuple[xr.Dataset, dict]:
        """From a given xarray dataset, returns a template that can be used to initialize an icechunk store.

        Template is written from the geobox, but uses the datasets coordinate names, and auto-detects the time dimension.

        This allows you to define a template that covers the entire geobox, without having to
        define an image collection that covers the entire geobox (which may be computationally inefficient).

        Args:
            ds: The xarray dataset to get the template and encoding for

        Returns:
            tuple[xr.Dataset, dict]: The template and encoding
        """

        y_dim, x_dim = spatial_dims(ds)

        has_time = "time" in ds.coords

        def template_for_dim(dim: str) -> xr.Dataset:
            dim_dtype = ds[dim].dtype

            time = ds.time.data if has_time else None

            chunk_array = xr_zeros(self.geobox, chunks=-1, dtype=dim_dtype, time=time)

            chunk_y_dim, chunk_x_dim = spatial_dims(chunk_array)

            chunk_array = chunk_array.rename({chunk_y_dim: y_dim, chunk_x_dim: x_dim})

            return chunk_array

        band_dictionary = {}
        coords = {}
        encoding = {}

        if has_time:
            coords["time"] = ("time", ds.time.data)

        encoding_chunks = self.chunk_size

        # default chunk size for time is always 1
        if has_time:
            encoding_chunks = (1, encoding_chunks[0], encoding_chunks[1])

        for band in list(ds.keys()):
            indexes = ("time", y_dim, x_dim) if has_time else (y_dim, x_dim)

            chunk_array = template_for_dim(band)

            # write the first bands spatial ref to the coords
            if "spatial_ref" not in coords:
                coords["spatial_ref"] = chunk_array.spatial_ref

            if y_dim not in coords:
                coords[y_dim] = chunk_array.coords[y_dim].data

            if x_dim not in coords:
                coords[x_dim] = chunk_array.coords[x_dim].data

            band_dictionary[band] = (
                indexes,
                chunk_array.data,
                {
                    "grid_mapping": "spatial_ref",
                },
            )

            encoding[band] = {
                "chunks": encoding_chunks,
            }

        template = xr.Dataset(band_dictionary, coords=coords)

        return template, encoding
