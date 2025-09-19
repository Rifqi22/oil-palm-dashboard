# fastapi_gee_oilpalm.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any
import geopandas as gpd
import json
import os
import time

from .core.gee_init import init_gee

# Earth Engine imports
import ee

# --- Configuration ---
ROI_PATH = "central_kalimantan.geojson"  # keep this file in repo root
RESOLUTION = 60  # meters for vectorization / scale
CACHE: Dict[str, Any] = {}

app = FastAPI(title="OilPalm GEE Visualizer")


class RunRequest(BaseModel):
    years: List[int]

import tempfile
import os

# Cross-platform temp dir
TMP_DIR = tempfile.gettempdir()




def load_roi_geojson(roi_path: str = ROI_PATH) -> Tuple[dict, ee.Geometry, dict]:
    """
    Load ROI from local GeoJSON (single feature or featurecollection) and return
    (geojson_dict, ee.Geometry (bounds), full featurecollection as ee.FeatureCollection)
    """
    if not os.path.exists(roi_path):
        raise FileNotFoundError(f"ROI file not found: {roi_path}")

    gdf = gpd.read_file(roi_path).to_crs("EPSG:4326")
    geojson = json.loads(gdf.to_json())

    # Create EE FeatureCollection
    fc = ee.FeatureCollection(geojson)
    geom = fc.geometry().bounds()  # bounding box geometry
    return geojson, geom, fc


def s1_median_gee(year: int, roi_fc: ee.FeatureCollection):
    """
    Build median VV and VH images for year (May-August) clipped to roi_fc.
    Returns an ee.Image with bands ['VV', 'VH'] (median values).
    """
    start = f"{year}-05-01"
    end = f"{year}-08-31"

    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(start, end)
        .filterBounds(roi_fc.geometry())
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING").Not())  # keep both passes optionally
        # .filter(ee.Filter.eq("resolution_meters", 10)) # not available as filter in some datasets
        .select(["VV", "VH"])
    )

    count = collection.size().getInfo()
    if count == 0:
        raise ValueError(f"No Sentinel-1 images found for year {year} in the ROI (May-Aug).")

    median = collection.median().clip(roi_fc.geometry())

    return median, int(count)


def compute_oil_palm_gee(year: int, roi_fc: ee.FeatureCollection, roi_geom: ee.Geometry) -> Dict[str, Any]:
    """
    Main logic in GEE:
      - compute median VV/VH (server-side)
      - compute RVI = (VV - VH) / (VV + VH)
      - apply thresholds: RVI > 0.675 and VH < 0.075
      - smooth mask (small morphological smoothing)
      - reduceToVectors and calculate area_ha
      - return GeoJSON FeatureCollection and summary
    """
    print(f"[GEE] Processing year {year}")
    start_t = time.time()

    median_img, n_images = s1_median_gee(year, roi_fc)
    print(f"[GEE] median built from {n_images} images")

    vv = median_img.select("VV")
    vh = median_img.select("VH")

    # RVI = (VV - VH) / (VV + VH)
    # Use ee.Image expressions to avoid division by zero
    rvi = vv.subtract(vh).divide(vv.add(vh).max(1e-12)).rename("RVI")

    # thresholds (converted from your scaled thresholds)
    rvi_thresh = 0.675
    vh_thresh = 0.075

    mask = rvi.gt(rvi_thresh).And(vh.lt(vh_thresh)).rename("oil_palm_mask")
    # Some smoothing: remove small speckles using focal mode (neighborhood)
    # Use a 3 pixel radius kernel (in meters) → kernel radius = approx RESOLUTION * 1.5
    kernel = ee.Kernel.square(radius=RESOLUTION, units="meters", normalize=False)
    # Perform a focal majority (mode) using reduceNeighborhood with reducer.mean then threshold > 0.5
    focal = mask.reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=kernel)
    smoothed = focal.gt(0.5).rename("oil_palm_smooth")

    # Convert to vector polygons
    # To avoid retrieving enormous vectors, we vectorize the clipped geometry only.
    vector_scale = RESOLUTION  # meters

    # reduceToVectors parameters
    vectors = (
        smoothed.selfMask()
        .reduceToVectors(
            geometry=roi_fc.geometry(),
            scale=vector_scale,
            geometryType="polygon",
            eightConnected=True,
            maxPixels=1e13,
            labelProperty="label",
        )
        .map(lambda f: f.set(
            "area_ha",
            ee.Number(f.geometry().area(maxError=RESOLUTION)).divide(10000)
        ))
    )


    # Get the result as GeoJSON (pull to client)
    # CAVEAT: getInfo() will block until server finishes and then return
    fc_json = vectors.getInfo()  # may be large for big AOIs
    elapsed = time.time() - start_t
    print(f"[GEE] Year {year} done in {elapsed:.1f}s | features: {len(fc_json.get('features', []))}")

    # compute total area
    features = fc_json.get("features", [])
    total_ha = sum([float(f["properties"].get("area_ha", 0)) for f in features])

    return {
        "year": year,
        "n_images": n_images,
        "geojson": fc_json,
        "area_ha": total_ha,
        "time_s": elapsed,
    }

@app.on_event("startup")
async def startup_event():
    try:
        init_gee()
    except Exception as e:
        print("GEE init failed:", e)

@app.post("/api/run")
def run(req: RunRequest):
    """
    Main API endpoint. Example request body:
      {"years":[2019,2020]}
    """
    key = ",".join(map(str, sorted(req.years)))
    if key in CACHE:
        return JSONResponse(content={"status": "cached", "key": key, "result": CACHE[key]["summary"]})

    # # Initialize EE
    # try:
    #     ee_init()
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))

    # Load ROI
    try:
        geojson, roi_geom, roi_fc = load_roi_geojson(ROI_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load ROI: {str(e)}")

    results = {}

    for year in req.years:
        try:
            print(f"[API] Computing year {year}")
            res = compute_oil_palm_gee(year, roi_fc, roi_geom)

            # Only save GeoJSON if we actually got one
            if "geojson" in res and res["geojson"].get("features"):
                fname = os.path.join(TMP_DIR, f"oil_palm_{year}.geojson")
                with open(fname, "w") as f:
                    json.dump(res["geojson"], f)
            else:
                fname = None

            results[year] = {
                "geojson_path": fname,
                "area_ha": res.get("area_ha", 0),
                "n_images": res.get("n_images", 0),
                "time_s": res.get("time_s", 0),
            }
        except Exception as e:
            results[year] = {"error": str(e)}


    CACHE[key] = {"summary": results}
    return JSONResponse(content={"status": "ok", "key": key, "result": results})


@app.get("/api/geojson/{year}")
def get_geojson(year: int):
    fname = f"/tmp/oil_palm_{year}.geojson"
    if not os.path.exists(fname):
        raise HTTPException(status_code=404, detail="GeoJSON not found; run /api/run first")
    return FileResponse(fname, media_type="application/geo+json")


@app.get("/api/status")
def status():
    return {"cached_keys": list(CACHE.keys())}


@app.get("/")
def test():
    return {"FASTAPI": "GEE WORKING"}
