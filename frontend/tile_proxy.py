from fastapi import FastAPI, Response
import httpx
import os
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:5501"] for tighter control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.environ.get("MGPF_CACHE_DIR", os.path.join(REPO_ROOT, "tile_cache"))
os.makedirs(CACHE_DIR, exist_ok=True)

# Import API_KEY from key.py with graceful fallbacks
try:
    from key import API_KEY
except ImportError:
    try:
        from frontend.key import API_KEY
    except ImportError:
        API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")

if not API_KEY:
    API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")

NASA_TILE_URL = (
    "https://api.nasa.gov/mars-wmts/catalog/"
    "Mars_Viking_MDIM21_ClrMosaic_global_232m/1.0.0/"
    "default/default028mm/{z}/{y}/{x}.jpg"
    f"?api_key={API_KEY}"
)

@app.get("/tiles/{z}/{x}/{y}.jpg")
async def proxy_tile(z: int, x: int, y: int):
    # ---- TILE BOUNDS GUARD (2:1 Aspect Ratio for Global Mars) ----
    max_y = 2 ** z
    max_x = 2 * max_y  # Global maps are usually 2:1 (360 deg width vs 180 deg height)
    
    if x < 0 or y < 0 or x >= max_x or y >= max_y:
        return Response(status_code=404) # 404 for out of bounds

    cache_path = os.path.join(CACHE_DIR, f"{z}_{x}_{y}.jpg")
    
    # Cache for 1 year (immutable)
    headers = {"Cache-Control": "public, max-age=31536000, immutable"}

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        with open(cache_path, "rb") as f:
            return Response(f.read(), media_type="image/jpeg", headers=headers)

    url = NASA_TILE_URL.format(z=z, x=x, y=y)

    timeout = httpx.Timeout(connect=10, read=60, write=10, pool=10)
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)

    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
        try:
            # Simple retry logic for 503s
            for attempt in range(3):
                try:
                    r = await client.get(url)
                    r.raise_for_status()
                    break # Success
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

    return Response(r.content, media_type="image/jpeg", headers=headers)

async def prefetch_tiles(max_zoom: int):
    import asyncio
    
    print(f" Starting SAFE pre-fetch for Zoom 0 to {max_zoom}...")
    print(f" Using API Key: {API_KEY[:5]}...{API_KEY[-5:]}")
    
    timeout = httpx.Timeout(connect=10, read=60, write=10, pool=10)
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
    
    # Track state across tasks
    state = {"stop": False, "total": 0, "errors": 0}

    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
        sem = asyncio.Semaphore(50) # MEGA-TURBO: Use 50 concurrent connections

        for z in range(max_zoom + 1):
            if state["stop"]: break
            
            max_y = 2 ** z
            max_x = 2 * max_y
            print(f"--> Scanning Zoom {z} ({max_x}x{max_y} tiles)...")
            
            tasks = []
            for y in range(max_y):
                if state["stop"]: break
                for x in range(max_x):
                    cache_path = os.path.join(CACHE_DIR, f"{z}_{x}_{y}.jpg")
                    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
                        continue # Skip existing
                        
                    tasks.append(download_tile(client, sem, z, x, y, state))
                    
                    if len(tasks) >= 500: # Larger batches (500)
                        await asyncio.gather(*tasks)
                        tasks = []
                        # No sleep in turbo mode

            if tasks:
                await asyncio.gather(*tasks)
            
    if state["stop"]:
        print("\n Pre-fetch HALTED due to API rate limiting (403/429).")
    else:
        print("\n Pre-fetch complete!")

async def download_tile(client, sem, z, x, y, state):
    if state["stop"]: return
    
    url = NASA_TILE_URL.format(z=z, x=x, y=y)
    cache_path = os.path.join(CACHE_DIR, f"{z}_{x}_{y}.jpg")
    
    async with sem:
        try:
            for attempt in range(3):
                try:
                    r = await client.get(url)
                    
                    if r.status_code == 403 or r.status_code == 429:
                        state["stop"] = True
                        return

                    r.raise_for_status()
                    
                    with open(cache_path, "wb") as f:
                        f.write(r.content)
                    state["total"] += 1
                    if state["total"] % 100 == 0:
                        print(f"    - Downloaded {state['total']} new tiles...")
                    return
                    
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                         return # Not found, legitimate skip
                    if e.response.status_code in [500, 502, 503, 504]:
                        await asyncio.sleep(2.0) # Wait longer on server errors
                        continue
                    raise e
        except Exception:
            state["errors"] += 1
            pass

if __name__ == "__main__":
    import uvicorn
    import argparse
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefetch", action="store_true", help="Download all tiles through max-zoom")
    parser.add_argument("--max-zoom", type=int, default=5, help="Max zoom level to prefetch")
    args = parser.parse_args()

    if args.prefetch:
        asyncio.run(prefetch_tiles(args.max_zoom))
    else:
        print("Starting Tile Proxy Server at http://127.0.0.1:8000")
        print("Ensure you have 'uvicorn' and 'httpx' installed.")
        uvicorn.run(app, host="127.0.0.1", port=8000)
