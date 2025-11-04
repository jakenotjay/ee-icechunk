"""Defines the chunk grid class, a mapping of a region of space to pixel coordinates and the writable chunks that cover it.

Definitions:

- Chunk: A group of pixels that are always written together, i.e. the smallest access size for a given zarr file.
- Region: A group of chunks that are written together, these are grouped together simply to enable quicker writes
- Chunk Grid: A mapping of a projected space to rows and columns of pixels, grouped by chunks and regions.

"""

import warnings
from typing import Any

import ee
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import spatial_dims, xr_zeros
from pyproj import CRS, Proj, Transformer
from shapely.geometry import Polygon, box


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
        x_dim: str = "lon",
        y_dim: str = "lat",
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

        self.x_dim = x_dim
        self.y_dim = y_dim

    @property
    def dst_crs(self) -> CRS:
        if isinstance(self.crs, Proj):
            return self.crs.crs

        return CRS.from_string(self.crs)

    def get_geobox(self) -> GeoBox:
        if not self.dst_crs.is_projected:
            # compute the geobox in 3857 first then convert to the target crs
            warnings.warn(
                "Target CRS is not projected, computing the geobox in EPSG:3857 and converting to target crs",
                stacklevel=2,
            )
            transformer = Transformer.from_crs(
                self.dst_crs, CRS.from_string("EPSG:3857"), always_xy=True
            )
            xmin, ymin = transformer.transform(self.xmin, self.ymin)
            xmax, ymax = transformer.transform(self.xmax, self.ymax)

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
        """For a given region (in pixels), returns the bounds in the destination CRS

        Args:
            region: The region to get the bounds for

        Returns:
            Polygon: The bounds of the region
        """

        # check region is within the datacube shape
        if region[self.x_dim].start < 0 or region[self.y_dim].start < 0:
            raise ValueError("Region is outside the datacube")
        if (
            region[self.x_dim].stop > self.datacube_shape[1]
            or region[self.y_dim].stop > self.datacube_shape[0]
        ):
            raise ValueError("Region is outside the datacube")

        affine = self.geobox.affine
        x_min, y_max = tuple[float, float](
            affine * (region[self.x_dim].start, region[self.y_dim].start)
        )
        x_max, y_min = tuple[float, float](
            affine * (region[self.x_dim].stop, region[self.y_dim].stop)
        )

        return box(x_min, y_min, x_max, y_max)

    def get_indexes_from_bounds(self, bounds: Polygon) -> dict[str, slice]:
        """Converts a bounds polygon to a slice of the datacube.

        Args:
            bounds: The bounds to convert to a region

        Returns:
            dict[str, slice]: The region
        """
        xmin, ymin, xmax, ymax = bounds.bounds
        x_start, y_start = tuple[float, float](~self.geobox.affine * (xmin, ymax))  # type: ignore[operator]
        x_end, y_end = tuple[float, float](~self.geobox.affine * (xmax, ymin))  # type: ignore[operator]
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
            self.x_dim: slice[int, int, Any](x_start, x_end),
            self.y_dim: slice[int, int, Any](y_start, y_end),
        }

    # TODO: should this instead be to the nearest chunk?
    # how do I create utilities that ease the burden of working with regions that don't cover the entire chunk grid
    def get_region_from_indexes(self, indexes: dict[str, slice]) -> dict[str, slice]:
        """Converts a slice of the datacube to a region in pixels i.e. it rounds up to the nearest region boundaries snapping to the nearest region."""
        x_start, x_end = indexes[self.x_dim].start, indexes[self.x_dim].stop
        y_start, y_end = indexes[self.y_dim].start, indexes[self.y_dim].stop

        x_start = x_start // self.region_size[0] * self.region_size[0]
        y_start = y_start // self.region_size[1] * self.region_size[1]

        x_end = (
            (x_end + self.region_size[0]) // self.region_size[0] * self.region_size[0]
        )
        y_end = (
            (y_end + self.region_size[1]) // self.region_size[1] * self.region_size[1]
        )

        return {
            self.x_dim: slice[int, int, Any](x_start, x_end),
            self.y_dim: slice[int, int, Any](y_start, y_end),
        }

    def get_all_regions(self, ds: xr.Dataset | None = None) -> list[dict[str, slice]]:
        """Returns a list of regions (in pixels) that cover the datacube, optionally using the dataset to determine the time dimension."""

        # TODO: this is not an eloquent interface for handling time
        # I wonder how we implement this better within chunk grid?
        has_time = ds is not None and "time" in ds.coords

        if has_time:
            time_indices = range(len(ds.time))

        x_step, y_step = self.region_size

        if x_step % self.chunk_size[1] != 0 or y_step % self.chunk_size[0] != 0:
            raise ValueError(
                "x_step and y_step must be integer multiples of the chunk size"
            )

        regions = []
        for y in range(0, self.datacube_shape[0], y_step):
            for x in range(0, self.datacube_shape[1], x_step):
                max_x = min(x + x_step, self.datacube_shape[1])
                max_y = min(y + y_step, self.datacube_shape[0])

                region = {
                    self.x_dim: slice(x, max_x),
                    self.y_dim: slice(y, max_y),
                }

                if not has_time:
                    regions.append(region)
                else:
                    for t in time_indices:
                        region_copy = region.copy()
                        region_copy["time"] = slice(t, t + 1)
                        regions.append(region_copy)

        return regions

    def get_regions_from_bounds(self, bounds: Polygon) -> list[dict[str, slice]]:
        """Converts region coordinates to pixel bounds, and then iterates over the nearest chunks to generate regions."""
        chunk_step_x, chunk_step_y = self.chunk_size
        region_step_x, region_step_y = self.region_size

        affine = self.geobox.affine

        xmin, ymin, xmax, ymax = bounds.bounds

        # start from top left hence flipped y
        x_start, y_start = tuple(~affine * (xmin, ymax))  # type: ignore
        x_end, y_end = tuple(~affine * (xmax, ymin))  # type: ignore

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
                    self.x_dim: slice(x, max_x),
                    self.y_dim: slice(y, max_y),
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

    def get_true_bounds(self) -> tuple[float, float, float, float]:
        """Returns the true bounds of the chunk grid in the destination CRS."""
        x_pix_max = self.datacube_shape[1]
        y_pix_max = self.datacube_shape[0]

        x_min, y_min = tuple(self.geobox.affine * (0, 0))
        x_max, y_max = tuple(self.geobox.affine * (x_pix_max, y_pix_max))
        return x_min, y_min, x_max, y_max

    def get_ee_bounds(self) -> ee.geometry.Geometry:
        """Returns an earth engine geometry object for the entire chunk grid, always in EPSG:4326."""

        x_min, y_min, x_max, y_max = self.get_true_bounds()

        if self.dst_crs.to_string() == "EPSG:4326":
            return ee.Geometry.BBox(x_min, y_min, x_max, y_max)

        transformer = Transformer.from_crs(
            self.dst_crs, CRS.from_string("EPSG:4326"), always_xy=True
        )
        xmin, ymin = transformer.transform(x_min, y_min)
        xmax, ymax = transformer.transform(x_max, y_max)

        return ee.Geometry.BBox(xmin, ymin, xmax, ymax)

    def get_region_ee_bounds(self, region: dict[str, slice]) -> ee.geometry.Geometry:
        """Returns an earth engine geometry object for a given region slice, always in EPSG:4326."""
        poly_box = self.get_bounds_from_region(region)
        x_min, y_min, x_max, y_max = poly_box.bounds
        return ee.Geometry.BBox(x_min, y_min, x_max, y_max)

    def get_all_region_ee_bounds(self) -> ee.featurecollection.FeatureCollection:
        """Returns an earth engine feature collection for all regions in the chunk grid."""
        all_regions = self.get_all_regions()
        all_regions_ee = [
            ee.Feature(self.get_region_ee_bounds(region)) for region in all_regions
        ]
        return ee.FeatureCollection(all_regions_ee)

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

        has_time = "time" in ds.coords

        def template_for_dim(dim: str) -> xr.Dataset:
            dim_dtype = ds[dim].dtype

            time = ds.time.data if has_time else None

            chunk_array = xr_zeros(self.geobox, chunks=-1, dtype=dim_dtype, time=time)

            chunk_y_dim, chunk_x_dim = spatial_dims(chunk_array)

            chunk_array = chunk_array.rename(
                {chunk_y_dim: self.y_dim, chunk_x_dim: self.x_dim}
            )

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
            indexes = (
                ("time", self.y_dim, self.x_dim)
                if has_time
                else (self.y_dim, self.x_dim)
            )

            chunk_array = template_for_dim(band)

            # write the first bands spatial ref to the coords
            if "spatial_ref" not in coords:
                coords["spatial_ref"] = chunk_array.spatial_ref

            if self.y_dim not in coords:
                coords[self.y_dim] = chunk_array.coords[self.y_dim].data

            if self.x_dim not in coords:
                coords[self.x_dim] = chunk_array.coords[self.x_dim].data

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
