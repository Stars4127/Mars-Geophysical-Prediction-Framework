from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import uvicorn
import time
import httpx
import os

CACHE_DIR = "tile_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

API_KEY = "esivxORdeiiLhojROVwck3hyOgO4weCOpUJyxUY5"

NASA_TILE_URL = (
    "https://api.nasa.gov/mars-wmts/catalog/"
    "Mars_Viking_MDIM21_ClrMosaic_global_232m/1.0.0/"
    "default/default028mm/{z}/{y}/{x}.jpg"
    f"?api_key={API_KEY}"
)


from engine import engine

app = FastAPI(title="MSPF API", description="Martian Landing Site Selection Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_methods=["*"],
    allow_headers=["*"],
)

class Weights(BaseModel):
    Water: float = 0.5
    Dust: float = 0.5
    Surface_Temp: float = 0.5
    Temp_Variation: float = 0.5
    Gravity: float = 0.5
    TI_Variance: float = 0.5
    Uncertainty_Penalty: float = 0.5

@app.get("/tiles/{z}/{x}/{y}.jpg")
async def proxy_tile(z: int, x: int, y: int):
    # EPSG:4326 base tiles are typically 2:1 ratio
    max_x = 2 ** (z + 1)
    max_y = 2 ** z
    
    if x < 0 or y < 0 or x >= max_x or y >= max_y:
        return Response(status_code=404)

    cache_path = os.path.join(CACHE_DIR, f"{z}_{x}_{y}.jpg")
    
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        with open(cache_path, "rb") as f:
            return Response(f.read(), media_type="image/jpeg", headers={"Cache-Control": "public, max-age=31536000"})

    url = NASA_TILE_URL.format(z=z, x=x, y=y)

    timeout = httpx.Timeout(connect=10, read=60, write=10, pool=10)
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)

    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
        try:
            for attempt in range(3):
                try:
                    r = await client.get(url)
                    r.raise_for_status()
                    break
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in [500, 502, 503, 504]:
                        if attempt < 2:
                            await __import__("asyncio").sleep(0.5)
                            continue
                    raise e
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                print(f"NASA API Error: {e.response.status_code} for {url}")
            return Response(status_code=404)
        except httpx.RequestError as e:
            print(f"Request Error: {e} for {url}")
            return Response(status_code=502)

    with open(cache_path, "wb") as f:
        f.write(r.content)

    return Response(r.content, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=31536000"})

@app.get("/")
async def root():
    return {"message": "MSPF API is Operational. Welcome to Mars."}

@app.post("/api/v1/calculate")
async def calculate_goodness(weights: Weights):
    start_time = time.time()
    try:
        results = engine.calculate_goodness(weights.dict())
        return {
            "status": "success",
            "compute_time": time.time() - start_time,
            "count": len(results),
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/point")
async def get_point_data(lat: float, lon: float):
    try:
        data = engine.get_nearest_point(lat, lon)
        return {
            "status": "success",
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/heatmap/{z}/{x}/{y}.png")
def get_heatmap_tile(z: int, x: int, y: int, 
                     water: float = 1.0, 
                     dust: float = 0.0, 
                     temp: float = 0.0, 
                     temp_var: float = 0.0, 
                     gravity: float = 0.0, 
                     ti_var: float = 0.0,
                     uncertainty: float = 0.5):
    weights = {
        "Water": water,
        "Dust": dust,
        "Surface_Temp": temp,
        "Temp_Variation": temp_var,
        "Gravity": gravity,
        "TI_Variance": ti_var,
        "Uncertainty_Penalty": uncertainty
    }
    
    try:
        img_bytes = engine.render_tile(z, x, y, weights)
        return Response(content=img_bytes, media_type="image/png")
    except Exception as e:
        print(f"Tile Error {z}/{x}/{y}: {e}")
        return Response(status_code=500)

@app.get("/api/v1/meta")
async def get_meta():
    return {
        "dataset": "MSPF 8ppd Global Extrapolation",
        "models": ["Water", "Dust", "Surface_Temp", "Temp_Variation", "Gravity", "TI_Variance"],
        "resolution": "8 ppd (pixels per degree)",
        "total_points": len(engine.df) if engine.df is not None else 0
    }

if __name__ == "__main__":
    # Explicitly ensure engine is loaded before starting the server
    print(" Pre-loading MSPF...")
    from engine import engine
    
    print(" Starting MSPF API Server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
