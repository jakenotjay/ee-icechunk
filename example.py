"""An example of how to use the ee-icechunk library to write the ERA-5 dataset to an icechunk store."""

import ee
import xarray as xr
import ee_ic

PROJECT = "jake-ee"

ee.Authenticate(auth_mode="gcloud")
ee.Initialize(project=PROJECT, opt_url="https://earthengine-highvolume.googleapis.com")


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

    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
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

    return ee.ImageCollection.fromImages(images)


start_date = ee.Date("2018-01-01")
print(start_date.getInfo())
count = 74  # july 2025
interval = 1
interval_units = "month"

target_crs = "EPSG:4326"
target_res = 10

xmin, ymin, xmax, ymax = (
    35.4607002239093,
    -16.12147572214461,
    35.82928740166918,
    -15.836284989543742,
)

grid = ee_ic.ChunkGrid(xmin, xmax, ymin, ymax, res=10, crs=target_crs)

ee_proj = grid.get_ee_projection()
print(ee_proj.getInfo())
ee_bounds = grid.get_ee_bounds()
print(ee_bounds.getInfo())
collection = get_sentinel_2_composites(
    ee_bounds, start_date, count, interval, interval_units
)

print(collection.first().getInfo())

ds = xr.open_dataset(collection, engine="ee", geometry=ee_bounds, projection=ee_proj)

print("loaded ds")
print(ds)

template, encoding = grid.get_template_and_encoding(ds)

print(template)
print(encoding)

# print(ds.attrs)

# n_y = 180 / 0.25
# n_x = 360 / 0.25

# chunk_grid = ee_ic.ChunkGrid.from_ee_proj(collection.first().projection(), (n_y, n_x))
