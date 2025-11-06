"""Defines the chunk grid class, a mapping of a region of space to pixel coordinates and the writable chunks that cover it.

Definitions:

- Chunk: A group of pixels that are always written together, i.e. the smallest access size for a given zarr file.
- Region: A group of chunks that are written together, these are grouped together simply to enable quicker writes
- Chunk Grid: A mapping of a projected space to rows and columns of pixels, grouped by chunks and regions.

"""

import warnings
from typing import Any

import ee
import shapely
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import spatial_dims, xr_zeros
from pyproj import CRS, Proj, Transformer
from shapely.geometry import Polygon, box

from ee_ic.utils import extract_dataset_config


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
        x_dim (str): The name of the x dimension
        y_dim (str): The name of the y dimension
        time_dim (str | None): The name of the time dimension (if present)
        time_coords (Any | None): The time coordinate values (if present)
        time_region-size (int): The size of time regions to write at once
        bands (dict[str, Any] | None): Mapping of band names to their dtypes
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
        time_dim: str | None = None,
        time_coords: Any | None = None,
        time_region_size: int = 1,
        bands: dict[str, Any] | None = None,
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

        self.time_dim = time_dim
        self.time_coords = time_coords
        self.time_region_size = time_region_size
        self.bands = bands

    def __repr__(self) -> str:
        return f"ChunkGrid(xmin={self.xmin}, xmax={self.xmax}, ymin={self.ymin}, ymax={self.ymax}, res={self.res}, crs={self.crs}, chunk_size={self.chunk_size}, region_size={self.region_size}, x_dim={self.x_dim}, y_dim={self.y_dim}, time_dim={self.time_dim}, time_coords={self.time_coords}, time_region_size={self.time_region_size}, bands={self.bands})"

    def configure_from_dataset(
        self, ds: xr.Dataset, relaxed: bool = False
    ) -> "ChunkGrid":
        """Configure time and band information from a dataset.

        This method is useful when the ChunkGrid is created before the dataset is loaded
        (e.g., when using the grid's projection and bounds to load Earth Engine data).

        Args:
            ds: The xarray dataset to extract configuration from
            relaxed: If True, use relaxed temporal dimension detection

        Returns:
            Self for method chaining
        """
        config = extract_dataset_config(ds, relaxed=relaxed)

        self.time_dim = config["time_dim"]
        self.time_coords = config["time_coords"]
        self.bands = config["bands"]

        return self

    @property
    def has_time(self) -> bool:
        return self.time_dim is not None and self.time_coords is not None

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
        x_start, y_start = ~self.geobox.affine * (xmin, ymax)  # pyright: ignore[reportOperatorIssue]
        x_end, y_end = ~self.geobox.affine * (xmax, ymin)  # pyright: ignore[reportOperatorIssue]
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
            self.x_dim: slice(x_start, x_end),
            self.y_dim: slice(y_start, y_end),
        }

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

        region = {
            self.x_dim: slice(x_start, x_end),
            self.y_dim: slice(y_start, y_end),
        }

        if self.has_time and self.time_dim in indexes:
            region[self.time_dim] = indexes[self.time_dim]

        return region

    def get_all_regions(self) -> list[dict[str, slice]]:
        """Returns a list of regions (in pixels) that cover the datacube.

        If time dimension is configured, returns spatiotemporal regions (one per time step per spatial region).
        Otherwise, returns spatial-only regions.

        Returns:
            list[dict[str, slice]]: List of region specifications as dicts of slices
        """
        x_step, y_step = self.region_size

        if x_step % self.chunk_size[1] != 0 or y_step % self.chunk_size[0] != 0:
            raise ValueError(
                "x_step and y_step must be integer multiples of the chunk size"
            )

        regions: list[dict[str, slice]] = []
        for y in range(0, self.datacube_shape[0], y_step):
            for x in range(0, self.datacube_shape[1], x_step):
                max_x = min(x + x_step, self.datacube_shape[1])
                max_y = min(y + y_step, self.datacube_shape[0])

                region = {
                    self.x_dim: slice(x, max_x),
                    self.y_dim: slice(y, max_y),
                }

                if not self.has_time:
                    regions.append(region)
                else:
                    time_indices = range(
                        0, len(self.time_coords), self.time_region_size
                    )  # pyright: ignore[reportArgumentType]
                    for t in time_indices:
                        region_copy = region.copy()
                        region_copy[self.time_dim] = slice(t, t + 1)  # pyright: ignore[reportArgumentType]
                        regions.append(region_copy)

        return regions

    def get_regions_from_bounds(self, bounds: Polygon) -> list[dict[str, slice]]:
        """Converts region coordinates to pixel bounds, and then iterates over the nearest chunks to generate regions."""
        aligned_bounds = self.get_grid_aligned_pixel_bounds(bounds)
        x_start, y_start, x_end, y_end = aligned_bounds

        region_step_x, region_step_y = self.region_size

        regions: list[dict[str, slice]] = []
        for y in range(y_start, y_end, region_step_y):
            for x in range(x_start, x_end, region_step_x):
                max_x = min(x + region_step_x, x_end, self.datacube_shape[1])
                max_y = min(y + region_step_y, y_end, self.datacube_shape[0])

                region = {
                    self.x_dim: slice(x, max_x),
                    self.y_dim: slice(y, max_y),
                }

                if not self.has_time:
                    regions.append(region)
                else:
                    time_indices = range(
                        0, len(self.time_coords), self.time_region_size
                    )  # pyright: ignore[reportArgumentType]
                    for t in time_indices:
                        region_copy = region.copy()
                        region_copy[self.time_dim] = slice(t, t + 1)  # pyright: ignore[reportArgumentType]
                        regions.append(region_copy)

        return regions

    def get_subregions_in_bounds(self, bounds: Polygon) -> list[dict[str, slice]]:
        """Converts region coordinates to pixel bounds of the subregion area, i.e. if you had
        a grid that only covered the bounds polygon, this method returns the pixel coordinates in local
        pixel coordinates rather than in your larger "global" space.

        If time dimension is configured, returns spatiotemporal regions.
        Otherwise, returns spatial-only regions.
        """
        x_start, y_start, x_end, y_end = self.get_grid_aligned_pixel_bounds(bounds)

        region_step_x, region_step_y = self.region_size

        local_width = x_end - x_start
        local_height = y_end - y_start

        regions: list[dict[str, slice]] = []
        for y in range(0, local_height, region_step_y):
            for x in range(0, local_width, region_step_x):
                max_x = min(x + region_step_x, local_width)
                max_y = min(y + region_step_y, local_height)

                region = {
                    self.x_dim: slice(x, max_x),
                    self.y_dim: slice(y, max_y),
                }

                if not self.has_time:
                    regions.append(region)
                else:
                    time_indices = range(
                        0, len(self.time_coords), self.time_region_size
                    )  # pyright: ignore[reportArgumentType]
                    for t in time_indices:
                        region_copy = region.copy()
                        region_copy[self.time_dim] = slice(t, t + 1)  # pyright: ignore[reportArgumentType]
                        regions.append(region_copy)

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

    def get_grid_aligned_pixel_bounds(
        self, intended_bounds: Polygon
    ) -> tuple[int, int, int, int]:
        """Given an arbritrary bounding box, align it to the grid boundaries, by rounding it to the nearest chunk grid boundaries."""

        chunk_step_x, chunk_step_y = self.chunk_size
        xmin, ymin, xmax, ymax = intended_bounds.bounds

        affine = self.geobox.affine
        # start from top left hence flipped y
        x_start, y_start = tuple(~affine * (xmin, ymax))  # pyright: ignore[reportOperatorIssue]
        x_end, y_end = tuple(~affine * (xmax, ymin))  # pyright: ignore[reportOperatorIssue]
        x_start, y_start = int(x_start), int(y_start)
        x_end, y_end = int(x_end), int(y_end)

        # floor to get the minimum chunk bounds
        x_start = x_start // chunk_step_x * chunk_step_x
        y_start = y_start // chunk_step_y * chunk_step_y

        # ceiling to get the maximum chunk bounds
        x_end = (x_end + chunk_step_x) // chunk_step_x * chunk_step_x
        y_end = (y_end + chunk_step_y) // chunk_step_y * chunk_step_y

        return x_start, y_start, x_end, y_end

    def get_grid_aligned_bounds(self, intended_bounds: Polygon) -> Polygon:
        """Given an arbritrary bounding box, align it to the grid boundaries, by rounding it to the nearest chunk grid boundaries."""
        x_start, y_start, x_end, y_end = self.get_grid_aligned_pixel_bounds(
            intended_bounds
        )

        affine = self.geobox.affine
        x_min, y_max = tuple(affine * (x_start, y_start))
        x_max, y_min = tuple(affine * (x_end, y_end))
        return box(x_min, y_min, x_max, y_max)

    def get_grid_aligned_ee_bounds(
        self, intended_bounds: Polygon
    ) -> ee.geometry.Geometry:
        x_min, y_min, x_max, y_max = self.get_grid_aligned_bounds(
            intended_bounds
        ).bounds
        return ee.Geometry.BBox(x_min, y_min, x_max, y_max)

    def get_ee_bounds_of_regions(
        self, regions: list[dict[str, slice]]
    ) -> ee.geometry.Geometry:
        """Returns an earth engine Bbox object for a list of regions."""
        all_bounds = [self.get_bounds_from_region(region) for region in regions]
        combined = shapely.union_all(all_bounds)
        x_min, y_min, x_max, y_max = combined.bounds
        return ee.Geometry.BBox(x_min, y_min, x_max, y_max)

    def get_all_region_ee_bounds(self) -> ee.featurecollection.FeatureCollection:
        """Returns an earth engine feature collection for all regions in the chunk grid."""
        all_regions = self.get_all_regions()
        all_regions_ee = [
            ee.Feature(self.get_region_ee_bounds(region)) for region in all_regions
        ]
        return ee.FeatureCollection(all_regions_ee)

    def get_template_and_encoding(self) -> tuple[xr.Dataset, dict[str, Any]]:
        """Returns a template and encoding that can be used to initialize an icechunk store.

        Template is written from the geobox and uses the configured band specifications,
        time dimension, and coordinate names.

        This allows you to define a template that covers the entire geobox, without having to
        define an image collection that covers the entire geobox (which may be computationally inefficient).

        Returns:
            tuple[xr.Dataset, dict]: The template and encoding

        Raises:
            ValueError: If bands configuration is not set
        """
        if self.bands is None:
            raise ValueError(
                "Bands configuration is required to generate template. "
                "Either call configure_from_dataset(ds) or set grid.bands manually."
            )

        def template_for_dim(dim_dtype: Any) -> xr.DataArray:
            time = self.time_coords if self.has_time else None

            chunk_array = xr_zeros(self.geobox, chunks=-1, dtype=dim_dtype, time=time)  # type: ignore

            s_dims = spatial_dims(chunk_array)

            if s_dims is None:
                raise ValueError("Spatial dimensions not found")

            chunk_y_dim, chunk_x_dim = s_dims

            chunk_array = chunk_array.rename(
                {chunk_y_dim: self.y_dim, chunk_x_dim: self.x_dim}
            )

            return chunk_array

        band_dictionary = {}
        coords = {}
        encoding = {}

        if self.has_time:
            coords[self.time_dim] = (self.time_dim, self.time_coords)

        encoding_chunks = self.chunk_size

        if self.has_time:
            encoding_chunks = (
                1,
                encoding_chunks[0],
                encoding_chunks[1],
            )

        for band, band_dtype in self.bands.items():
            indexes = (
                (self.time_dim, self.y_dim, self.x_dim)
                if self.has_time
                else (self.y_dim, self.x_dim)
            )

            chunk_array = template_for_dim(band_dtype)

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
