# ee-icechunk
A helper library for typical workflows with Earth Engine (via XEE) and icechunk for distributed writes at scale

## Key concepts and disambiguation
- Chunks: The lowest level of granularity for a given zarr file, array chunks are stored as a part of a pixel grid
- Region (needs a rename): A group of chunks that are adjacent to one another in the grid space, i.e. either next to each other in one of the cardinal directions or time contiguous
    - Regions can be as large as a datacube or the size of a single chunk, its defined as a way to maximise IO operations.
- Chunk Grid: The chunk grid is a mapping of a region of space to pixel coordinates and the writable chunks that cover it, its defined by its projection, bounds, resolution and chunk size.
    - The chunk grid is the primary way of defining an icechunk store (datacube), and allows for a simpler way of defining the pixel grids you expect to write to, matching the chunks from the original data.
- Subregion (needs an even bigger rename): Subregions are local pixel coordinates for a smaller subset of the entire datacube (see below for description of the primary use case), i.e. if we have a datacube thats 100,000 x 100,000 pixels, a subregion could be the central 10,000 x 10,000 pixels, this would have coordinates 0,0 to 9999,9999 in the subregion, but in the global coordinate system (the region) it would have coordinates 45000,45000 to 54999,54999.
- High Volume Endpoint: A special endpoint for Earth Engine that allows for larger concurrent operations, it can be used to write to icechunk stores, alongside XEE which provides a connector to xarray.

## Typical usage
See the global_grid_locally_defined.py script for a typical usage of the library, this is a script that writes a datacube to a global grid with a locally defined AOI.

Key points to note:
- use the recommended_io_chunks method to get the optimal io_chunks for your datacube (when n_time is known)
- use this as your region size and time region size
- the grid is defined globally, this means you'll need to use the grid aligned projection when pulling the data from earth engine
- once you have got your datacube, you can use the configure_from_dataset method to configure the grid with the time, and band information of the datacube if you haven't already done so (in production settings this should be known ahead of time but for first time initialisation, it'll allow for an easy setup)
- furthermore, the initial bounds you or a user give are very unlikely to properly align with the grid boundaries, and rather than writing a chunk that is only half full, we use the get_grid_aligned_bounds method to buffer the bounds to the nearest chunk boundaries
- you must then use both the get_regions_from_bounds (for knowing the indexes within the global datacube) and the get_subregions_in_bounds (for knowing the local pixel coordinates of the user requested region) to properly select and define the the regions to write to
    - these methods will return the same number of regions, with the same shapes

## Notes on io_chunks, choosing a region size and the number of threads (per worker)

### io_chunks
**Based on messing about with various bits, its actually not worth configuring the below, it doesn't make a difference + I don't have an understanding of the underlying bandwidth/request volume that can be made at the moment**

- xee has a concept of io_chunks, these are the number of pixels in a single request that xee will make to earth engine when calling [compute pixels](https://developers.google.com/earth-engine/reference/rest/v1/projects.image/computePixels), this is required because there is a limit of 48MBs per request
- this is done on a per-band basis, meaning each is "computed" lazily as a separate request and can include many time steps in a single request
- xee attempts to automatically determine the optimal io_chunks using the [_auto_chunk](https://github.com/google/Xee/blob/e82ecb4d25b2f1ce05d8cd3bf859c43862f1634e/xee/ext.py#L349) method which takes the size of the datatype of the band in bytes (e.g. float32 is 4 bytes + 1 byte for the mask), and returns a dictionary of "index", "width" and "height" values, index being the number of time steps to request at once
- you can override this by passing a custom `io_chunks` dictionary to the xarray `open_dataset` method (which gets passed on the earth engine store), for example passing -1 would request the entire datacube in a single request
- The current method however, massively overprovisions bandwidth for the temporal dimension by assuming that half of the budget is allocated to the temporal dimension, meaning it typically returns an io_chunks dictionary of {"index": 48, "width" 256, "height": 256}, no matter the input
- In order to maximise throughput, I've implemented a custom method in `ee_ic.utils.recommend_io_chunks` which takes one extra parameter, the number of time steps in the datacube, which we will always know when writing the datacube, this is used to then compute the optimal io_chunks for the datacube by ensuring that bandwidth for the temporal dimension is exact, and then providing the remaining bandwidth to the width and height dimensions
- There's one key point, which is that maths is all done in binary logarithmic space, meaning that the width and height dimensions are always rounded to powers of two e.g. 256, 512, 1024 etc. meaning there can be a lot of bandwidth left over (2048 x 2048 * 5 bytes is ~20MBs, so you can't fit much time into a single request for a larger geographic region)
- However, this is a sacrifice made for the sake of conforming to a standard grid space, meaning that chunks are always the same size and normally powers of two
- There's a possibility that you could instead write chunks that are rounded to powers of 10, but would require smaller chunks to be written and then you likely suffer IO overheads from the smaller chunks
- Generally speaking, the optimal width and height is 1024 x 1024 (x 5bytes = 5.24MBs) for float32, with the available time bandwidth being equivalent to 9 time steps (this is pretty great actually as we typically run 2017 - 2025 anually)
- A final note, the method also takes a `min_width` and `min_height` parameter, this is useful for temporally dense datasets, where it'll be pretty unhelpful if the io_chunks are 64 x 64 or something similar just to fit the entire temporal dimension
- I would recommend that min_width and min_height are set to your chunk size

### Selecting a region size
- I would recommend that you select a region size that matches the io_chunks width, height and temporal depth
- Because we write all bands at once, this means that each task submitted to dask, will result in n_bands requests to earth engine (unless the temporal depth is greater than the io_chunks index, in which case it will be n_bands * (n_time // io_depth))

### Selecting the number of threads (per worker)
- following on from the above two sections, it now becomes trivial to maximise throughput for a given max concurrent requests
- each task will result in n_bands * (n_time // io_depth) requests to ee, in most cases n_time// io_depth will be 1, therefore we assume that each task will result in n_bands requests to ee
- max_concurrent_requests // n_bands will give you number of concurrent tasks that can be submitted to ee
- in other words ceiling(max_concurrent_requests / n_bands / n_workers) will slightly overprovision, but given that xee auto manages failures, this will allow us to saturate the throughput

### Untested considerations
- For earth engine workflows with temporal modelling e.g. a harmonic analysis or ccdc, you're almost always going to get better performance by writing the entire temporal dimension at once due to the computation of the last step requiring the entire temporal dimension
- Given that the number of bytes per pixel is a considerable factor in the io_chunks calculation, a higher throughput may be achieved by compressing the data on the fly, this could be achieved by converting float32 to uint16 or uint8 before writing to icechunk as its likely that the multiplication is very quick, however this may have the unintended consequence of massively slowing down each request as you request more pixels (more computation per request) so its a tricky balance
    - one could assume though that increasing the throughput by 100% may only result in each request being 50% slower, which is still significant
- the behaviour of the high volume endpoint is quite unknown to me, if you have a lower concurrent request limit you may still prefer the normal online endpoint
