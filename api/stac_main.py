from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from os import cpu_count
import tempfile
import os
import json
import time
import math

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import transform
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import shapes
from planetary_computer import sign_inplace
from pystac_client import Client
from skimage.segmentation import slic
from scipy import ndimage
from skimage.exposure import rescale_intensity
from shapely.geometry import shape, mapping


# --- Configuration ---
ROI_PATH = "central_kalimantan.geojson"  # keep this file in the repo root (next to app)
CATALOG = "https://planetarycomputer.microsoft.com/api/stac/v1/"
COLLECTION = "sentinel-1-rtc"
BANDS = ["vv", "vh"]
CPU_COUNT = max(1, cpu_count() - 1)
RESOLUTION = 60  # meters
TARGET_CRS = "EPSG:4326"

# simple in-memory cache so repeated requests don't recompute
CACHE: Dict[str, Any] = {}

app = FastAPI(title="OilPalm S1 Visualizer")


class RunRequest(BaseModel):
    years: List[int]


# Utility: load ROI and compute bbox and output shape
def load_roi(roi_path: str = ROI_PATH) -> Tuple[gpd.GeoDataFrame, Tuple[float, float, float, float], Tuple[int, int], transform.Affine]:
    roi = gpd.read_file(roi_path).to_crs(TARGET_CRS)
    minx, miny, maxx, maxy = tuple(roi.total_bounds)

    width = int(abs(minx - maxx) * 111_000 / RESOLUTION)
    height = int(abs(miny - maxy) * 111_000 / RESOLUTION)
    out_shape = (height, width)
    dst_transform = transform.from_bounds(minx, miny, maxx, maxy, width, height)
    return roi, (minx, miny, maxx, maxy), out_shape, dst_transform


# Warp / read and resample single asset (vsicurl path expected)
def warp_image(src: str, bbox: Tuple[float, float, float, float], out_shape: Tuple[int, int]) -> np.ma.MaskedArray:
    # Use rasterio to open the remote file and reproject/resample into the target bounds/crs
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    tmpfile.close()

    minx, miny, maxx, maxy = bbox
    width, height = out_shape[1], out_shape[0]
    dst_transform = transform.from_bounds(minx, miny, maxx, maxy, width, height)

    src_path = f"/vsicurl/{src}"
    try:
        with rio.open(src_path) as s:
            # read first band
            src_array = s.read(1)
            src_crs = s.crs
            src_transform = s.transform

            # prepare destination
            dst_meta = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': 1,
                'dtype': 'float32',
                'crs': TARGET_CRS,
                'transform': dst_transform,
            }

            dst = np.zeros((height, width), dtype='float32')

            reproject(
                source=src_array,
                destination=dst,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.lanczos,
            )

            # mask zeros / nodata
            dst = np.ma.masked_equal(dst, 0)
            # scale similar to original notebook (multiply by 1e4 and cast)
            scaled = (dst * 1e4).astype('uint16')
            return scaled
    except Exception as e:
        raise


def median_image(images: List[np.ma.MaskedArray]) -> np.ma.MaskedArray:
    stack = np.ma.stack(images, dtype='uint16')
    median = np.ma.median(stack, axis=0, overwrite_input=True).astype('uint16')
    np.ma.set_fill_value(median, 0)
    return median


# Tile bbox into smaller sub-bboxes
def tile_bbox(bbox, tile_size_deg=0.2):  # ~20km at equator
    minx, miny, maxx, maxy = bbox
    x_steps = math.ceil((maxx - minx) / tile_size_deg)
    y_steps = math.ceil((maxy - miny) / tile_size_deg)
    tiles = []
    for i in range(x_steps):
        for j in range(y_steps):
            x0 = minx + i * tile_size_deg
            x1 = min(minx + (i+1) * tile_size_deg, maxx)
            y0 = miny + j * tile_size_deg
            y1 = min(miny + (j+1) * tile_size_deg, maxy)
            tiles.append((x0, y0, x1, y1))
    return tiles


def human_size(num_bytes: int) -> str:
    """Convert bytes into human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"

# Optimized median function with logging
def s1_median_optimized(year: int, bbox, out_shape, resolution: int = 30):
    print(f"[S1 MEDIAN] 🚀 Processing year {year}")

    # Step 1: Find items
    client = Client.open(CATALOG, modifier=sign_inplace)
    items = list(client.search(
        collections=[COLLECTION],
        datetime=(f"{year}-05", f"{year}-08"),
        bbox=bbox
    ).items())
    print(f"[S1 MEDIAN] ✓ STAC search done | Items found: {len(items)}")

    if not items:
        raise ValueError(f"[S1 MEDIAN] ❌ No items found for {year}")

    # Step 2: Tile the bbox
    tiles = tile_bbox(bbox, tile_size_deg=0.2)  # tweak size
    vv_images, vh_images = [], []

    for t, tile in enumerate(tiles, 1):
        print(f"[S1 MEDIAN] → Processing tile {t}/{len(tiles)} {tile}")

        with ThreadPoolExecutor(CPU_COUNT) as ex:
            vv_jobs, vh_jobs = [], []
            for item in items:
                for band in BANDS:
                    path = item.assets[band].href
                    if band == "vv":
                        vv_jobs.append(ex.submit(warp_image, path, bbox, out_shape))
                    else:
                        vh_jobs.append(ex.submit(warp_image, path, bbox, out_shape))

            # collect VV
            for j in vv_jobs:
                arr = j.result()
                if arr is not None:
                    print(f"[S1 MEDIAN]   ✓ VV image loaded | Shape: {arr.shape} | Size: {human_size(arr.nbytes)}")
                    vv_images.append(arr)

            # collect VH
            for j in vh_jobs:
                arr = j.result()
                if arr is not None:
                    print(f"[S1 MEDIAN]   ✓ VH image loaded | Shape: {arr.shape} | Size: {human_size(arr.nbytes)}")
                    vh_images.append(arr)

    if not vv_images or not vh_images:
        raise ValueError(f"[S1 MEDIAN] ❌ No VV/VH images processed")

    print(f"[S1 MEDIAN] → Computing medians for {len(vv_images)} VV and {len(vh_images)} VH images...")
    vv_median = median_image(vv_images)
    vh_median = median_image(vh_images)
    print(f"[S1 MEDIAN] ✅ Done {year} | VV: {len(vv_images)}, VH: {len(vh_images)} | Median shape: {vv_median.shape}, size: {human_size(vv_median.nbytes)}")

    return vv_median, vh_median



# Produce oil palm raster and polygon for a year
def compute_oil_palm_for_year(
    year: int,
    bbox: Tuple[float, float, float, float],
    out_shape: Tuple[int, int],
    dst_transform
) -> Dict[str, Any]:
    print(f"[COMPUTE OIL PALM] 🚀 Processing year {year}")

    # Step 1: Median calculation
    print(f"[COMPUTE OIL PALM] → Calculating median Sentinel-1 composites...")
    vv, vh = s1_median_optimized(year, bbox, out_shape)
    print(f"[COMPUTE OIL PALM] ✓ Median done | VV shape: {vv.shape}, VH shape: {vh.shape}")

    # Step 2: RVI calculation
    print(f"[COMPUTE OIL PALM] → Calculating RVI...")
    rvi = ((vv / 1e4 - vh / 1e4) / (vv / 1e4 + vh / 1e4) * 1e4).astype('int16')
    print(f"[COMPUTE OIL PALM] ✓ RVI done | RVI min: {rvi.min()}, max: {rvi.max()}")

    # Step 3: Composite for segmentation
    print(f"[COMPUTE OIL PALM] → Building composite stack...")
    composite = np.dstack([
        rescale_intensity(vv, in_range=(1000, 4000), out_range=(0, 1)),
        rescale_intensity(vh, in_range=(0, 1000), out_range=(0, 1)),
        rescale_intensity(rvi, in_range=(5000, 10000), out_range=(0, 1)),
    ])
    print(f"[COMPUTE OIL PALM] ✓ Composite done | Shape: {composite.shape}")

    # Step 4: Segmentation
    print(f"[COMPUTE OIL PALM] → Running SLIC segmentation...")
    segments = slic(composite, n_segments=2000, sigma=5, compactness=5)
    unique = np.unique(segments)
    print(f"[COMPUTE OIL PALM] ✓ Segmentation done | Segments: {len(unique)}")

    # Step 5: Segment statistics
    print(f"[COMPUTE OIL PALM] → Calculating mean VV, VH, RVI per segment...")
    mean_vv = ndimage.mean(vv, labels=segments, index=unique)
    mean_vh = ndimage.mean(vh, labels=segments, index=unique)
    mean_rvi = ndimage.mean(rvi, labels=segments, index=unique)
    print(f"[COMPUTE OIL PALM] ✓ Segment stats done")

    # Map segment IDs to indices
    id_to_idx = np.zeros(segments.max() + 1, dtype=int)
    id_to_idx[unique] = np.arange(len(unique))

    # Replace with segment means
    segments_vv = mean_vv[id_to_idx[segments]]
    segments_vh = mean_vh[id_to_idx[segments]]
    segments_rvi = mean_rvi[id_to_idx[segments]]

    # Step 6: Oil palm mask
    print(f"[COMPUTE OIL PALM] → Applying oil palm thresholding...")
    oil_palm = (segments_rvi > 6750) & (segments_vh < 750)
    print(f"[COMPUTE OIL PALM] ✓ Oil palm mask done | Palm pixels: {np.sum(oil_palm)}")

    # Step 7: Polygonization
    print(f"[COMPUTE OIL PALM] → Polygonizing oil palm areas...")
    mask = oil_palm.astype('uint8')
    polygons = []
    for geom, val in shapes(mask, mask=mask.astype(bool), transform=dst_transform):
        if val == 1:
            polygons.append(mapping(shape(geom)))
    print(f"[COMPUTE OIL PALM] ✓ Polygonization done | Polygons: {len(polygons)}")

    # Step 8: Area calculation
    area_ha = int(np.sum(mask) * 900 / 10000)  # 30m pixel → 0.09 ha
    print(f"[COMPUTE OIL PALM] ✓ Area calculated | {area_ha} ha")

    return {
        'year': year,
        'oil_palm_raster': mask,  # not serialized
        'polygons': polygons,
        'area_ha': area_ha,
    }

@app.post('/api/run')
def run(req: RunRequest, background_tasks: BackgroundTasks):
    print('running request')
    # simple cache key
    key = ",".join(map(str, sorted(req.years)))
    if key in CACHE:
        return JSONResponse(content={"status": "cached", "key": key, "result": CACHE[key]['summary']})

    print('loading roi')
    roi, bbox, out_shape, dst_transform = load_roi()
    print('loading roi done, result:', roi, bbox, out_shape)

    results = {}
    for year in req.years:
        try:
            print('computing oil palm for year', year)
            res = compute_oil_palm_for_year(year, bbox, out_shape, dst_transform)
            print('computing done with result:', res)
            # save geojson file to /tmp
            fname = f"/tmp/oil_palm_{year}.geojson"
            geo = {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'properties': {'year': year},
                        'geometry': poly,
                    }
                    for poly in res['polygons']
                ],
            }
            with open(fname, 'w') as f:
                json.dump(geo, f)

            results[year] = {
                'geojson_path': fname,
                'area_ha': res['area_ha'],
            }
        except Exception as e:
            results[year] = {'error': str(e)}

    CACHE[key] = {'summary': results}
    return JSONResponse(content={'status': 'ok', 'key': key, 'result': results})

@app.get("/")
def test():
    return {"FASTAPI": "WORKING"}

@app.get('/api/geojson/{year}')
def get_geojson(year: int):
    fname = f"/tmp/oil_palm_{year}.geojson"
    if not os.path.exists(fname):
        raise HTTPException(status_code=404, detail="GeoJSON not found; run /api/run first")
    return FileResponse(fname, media_type='application/geo+json')


@app.get('/api/status')
def status():
    return {'cached_keys': list(CACHE.keys())}
