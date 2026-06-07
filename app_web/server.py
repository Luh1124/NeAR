import asyncio
import os
import shutil
import uuid
import base64
import secrets
from pathlib import Path
from typing import Optional, Any
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
except Exception:
    pass

try:
    from tqdm import tqdm
    tqdm(disable=True, total=0)
except Exception:
    pass

import app_web.config as config
from app_web.config import (
    APP_DIR,
    CACHE_DIR,
    DEFAULT_SLAT,
    DEFAULT_HDRI,
    MOCK_MODE,
)
from app_web.pipelines import string_to_color
from app_web.worker import (
    gpu_lock,
    pipeline_manager,
    run_in_gpu_executor,
    generate_3d_asset_task,
    generate_style_transfer_task,
    export_glb_task,
)
from app_web.pipelines import save_slat_npz

app = FastAPI(title="NeAR Web App")

# Authentication Helpers & Middleware
def check_credentials(auth_cookie: Optional[str]) -> bool:
    if not config.args or not config.args.username or not config.args.password:
        return True
    if not auth_cookie:
        return False
    try:
        decoded = base64.b64decode(auth_cookie.encode("ascii")).decode("utf-8")
        username, password = decoded.split(":", 1)
        return secrets.compare_digest(username, config.args.username) and secrets.compare_digest(password, config.args.password)
    except Exception:
        return False

@app.middleware("http")
async def check_auth_middleware(request: Request, call_next):
    if config.args and config.args.username and config.args.password:
        if request.url.path in ("/login", "/login/"):
            return await call_next(request)
        
        auth_cookie = request.cookies.get("near_auth")
        if not check_credentials(auth_cookie):
            if request.url.path in ("/", "/view_glb", "/view_glb/"):
                return RedirectResponse(url="/login")
            return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=401)
            
    return await call_next(request)

class LoginRequest(BaseModel):
    username: str
    password: str

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    template_path = Path(__file__).resolve().parent / "templates" / "login.html"
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Login template not found.</h1>")

@app.post("/login")
async def login(data: LoginRequest):
    if config.args and config.args.username and config.args.password:
        if secrets.compare_digest(data.username, config.args.username) and secrets.compare_digest(data.password, config.args.password):
            return JSONResponse({"status": "success"})
    return JSONResponse({"status": "error", "message": "Invalid credentials"}, status_code=401)

def convert_exrs_to_hdrs():
    """Convert EXR files in assets/hdris to HDR format for WebGL frontend compat."""
    hdri_dir = APP_DIR / "assets" / "hdris"
    if not hdri_dir.exists():
        return
    
    import cv2
    import os
    # Temporarily set environment variable for OpenCV EXR support
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    
    print("[Startup] Scanning for EXR to HDR conversions in assets/hdris...")
    for f in hdri_dir.glob("*.exr"):
        hdr_path = f.with_suffix(".hdr")
        if not hdr_path.exists():
            try:
                print(f"[Startup] Converting {f.name} to HDR...")
                img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    # Write as Radiance HDR format
                    cv2.imwrite(str(hdr_path), img)
                    print(f"[Startup] Successfully wrote {hdr_path.name}")
                else:
                    print(f"[Startup] Warning: Failed to read {f.name}")
            except Exception as e:
                print(f"[Startup] Error converting {f.name}: {e}")


@app.on_event("startup")
async def startup_event():
    # Run the EXR -> HDR conversion on startup
    try:
        convert_exrs_to_hdrs()
    except Exception as e:
        print(f"[Startup] Failed to run EXR to HDR converter: {e}")

    if config.args and config.args.share:
        from gradio.networking import setup_tunnel
        loop = asyncio.get_running_loop()
        try:
            share_url = await loop.run_in_executor(
                None,
                lambda: setup_tunnel(
                    local_host="127.0.0.1",
                    local_port=config.args.port,
                    share_token=uuid.uuid4().hex[:16],  # Randomized share token to prevent proxy reuse name collisions
                    share_server_address=None,
                    share_server_tls_certificate=None
                )
            )
            print("\n" + "="*80)
            print(f"🎉 Gradio Share Tunnel successfully established!")
            print(f"🔗 Public URL: \033[94m\033[4m{share_url}\033[0m")
            print("="*80 + "\n", flush=True)
        except Exception as e:
            print(f"\n❌ Failed to establish Gradio Share Tunnel: {e}\n", flush=True)

# Mount static files
app.mount("/assets", StaticFiles(directory=str(APP_DIR / "assets")), name="assets")
app.mount("/tmp_web_app", StaticFiles(directory=str(CACHE_DIR)), name="tmp_web_app")


def get_available_slats(default_path: Optional[str] = None) -> list[dict[str, str]]:
    """Scan assets/example_slats and tmp_web_app for available SLaT .npz models."""
    slat_dir = APP_DIR / "assets/example_slats"
    slats = []
    
    # Check default_path
    if default_path and Path(default_path).is_file():
        default_path_obj = Path(default_path).resolve()
        slats.append({"name": f"{default_path_obj.stem} (Custom)", "path": str(default_path_obj)})
            
    if slat_dir.exists():
        for f in sorted(slat_dir.glob("*.npz")):
            # Avoid duplicating
            if not any(s["path"] == str(f) for s in slats):
                slats.append({"name": f.stem, "path": str(f)})
                
    # Also scan tmp_web_app for newly generated SLaTs
    if CACHE_DIR.exists():
        for f in sorted(CACHE_DIR.glob("**/generated_slat.npz")):
            slats.append({"name": f"Generated ({f.parent.name})", "path": str(f)})
            
    # Fallback if empty
    if not slats:
        slats.append({"name": "Default Model", "path": str(DEFAULT_SLAT)})
    return slats


def get_available_hdris(default_path: Optional[str] = None) -> list[dict[str, str]]:
    """Scan assets/hdris for available EXR environment maps."""
    hdri_dir = APP_DIR / "assets/hdris"
    hdris = []
    
    if default_path and Path(default_path).is_file():
        default_path_obj = Path(default_path).resolve()
        hdris.append({"name": f"{default_path_obj.stem} (Custom)", "path": str(default_path_obj)})
            
    if hdri_dir.exists():
        for f in sorted(hdri_dir.glob("*.exr")):
            if not any(h["path"] == str(f) for h in hdris):
                hdris.append({"name": f.stem, "path": str(f)})
            
    if not hdris:
        hdris.append({"name": "Default Studio", "path": str(DEFAULT_HDRI)})
    return hdris


def get_available_example_images() -> list[dict[str, str]]:
    """Scan assets/example_image for available example PNG/JPG images."""
    img_dir = APP_DIR / "assets/example_image"
    images = []
    if img_dir.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for f in img_dir.glob(ext):
                images.append({
                    "name": f.stem,
                    "path": f"assets/example_image/{f.name}"
                })
    return sorted(images, key=lambda x: x["name"])


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page application."""
    template_path = Path(__file__).resolve().parent / "templates" / "index.html"
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Templates not found. Please run template setup.</h1>")


@app.get("/view_glb", response_class=HTMLResponse)
async def view_glb_page():
    """Serve standalone GLB viewer page."""
    template_path = Path(__file__).resolve().parent / "templates" / "view_glb.html"
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>view_glb.html template not found.</h1>")


@app.get("/api/list_glbs")
async def list_glbs():
    """List all available GLB files in cache recursively."""
    glbs = []
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("**/*.glb"):
            glbs.append({
                "name": f"{f.parent.name}/{f.name}",
                "url": f"/tmp_web_app/{f.relative_to(CACHE_DIR)}"
            })
    return JSONResponse({"glbs": sorted(glbs, key=lambda x: x["name"])})


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image to start the 3D generation pipeline."""
    try:
        # Create a unique session directory
        session_id = str(uuid.uuid4())[:8]
        session_dir = CACHE_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = session_dir / "input_uploaded.png"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return JSONResponse({
            "status": "success",
            "session_id": session_id,
            "image_path": f"tmp_web_app/{session_id}/input_uploaded.png"
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/upload_style_transfer")
async def upload_style_transfer(
    geo_file: UploadFile = File(...),
    style_file: UploadFile = File(...)
):
    """Upload both geo and style images for Style Transfer 3D generation."""
    try:
        session_id = str(uuid.uuid4())[:8]
        session_dir = CACHE_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        geo_path = session_dir / "geo_image.png"
        style_path = session_dir / "style_image.png"
        
        with open(geo_path, "wb") as buffer:
            shutil.copyfileobj(geo_file.file, buffer)
            
        with open(style_path, "wb") as buffer:
            shutil.copyfileobj(style_file.file, buffer)
            
        return JSONResponse({
            "status": "success",
            "session_id": session_id,
            "geo_image_path": f"tmp_web_app/{session_id}/geo_image.png",
            "style_image_path": f"tmp_web_app/{session_id}/style_image.png"
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time rendering, state synchronization, and background task WebSocket."""
    if config.args and config.args.username and config.args.password:
        auth_cookie = websocket.cookies.get("near_auth")
        if not check_credentials(auth_cookie):
            auth_header = websocket.headers.get("authorization")
            is_header_valid = False
            if auth_header and auth_header.startswith("Basic "):
                try:
                    decoded = base64.b64decode(auth_header[6:].encode("ascii")).decode("utf-8")
                    username, password = decoded.split(":", 1)
                    if secrets.compare_digest(username, config.args.username) and secrets.compare_digest(password, config.args.password):
                        is_header_valid = True
                except Exception:
                    pass
            if not is_header_valid:
                await websocket.accept()
                await websocket.send_json({"type": "error", "message": "Unauthorized WebSocket connection"})
                await websocket.close(code=3000, reason="Unauthorized")
                return

    await websocket.accept()
    print("[WebSocket] Client connected")

    # Initial state
    current_slat_path = config.args.slat if config.args and config.args.slat else str(DEFAULT_SLAT)
    current_hdri_path = config.args.hdri if config.args and config.args.hdri else str(DEFAULT_HDRI)
    
    # Expose actual views supported by OpenColorIO ToneMapper dynamically
    try:
        from simple_ocio.tone_mapper import ToneMapper
        available_tm = ToneMapper().available_views
    except Exception:
        available_tm = ["AgX", "Standard", "Filmic", "Raw"]
    tone_default = "AgX" if "AgX" in available_tm else available_tm[0]

    # Send initial configuration to client
    init_payload = {
        "type": "init",
        "tone_mappers": available_tm,
        "default_tone_mapper": tone_default,
        "drag_res": 256,
        "idle_res": 1024,
        "yaw": 0.0,
        "pitch": 0.0,
        "radius": 2.0,
        "fov": 40.0,
        "mock_mode": MOCK_MODE,
        "slats": get_available_slats(current_slat_path),
        "hdris": get_available_hdris(current_hdri_path),
        "current_slat": current_slat_path,
        "current_hdri": current_hdri_path,
        "example_images": get_available_example_images(),
        "super_resolve": True,
    }
    await websocket.send_json(init_payload)

    # State variables for this connection
    client_state = {
        "yaw": 0.0,
        "pitch": 0.0,
        "radius": 2.0,
        "fov": 40.0,
        "is_dragging": False,
        "hdri_rot_deg": 0.0,
        "tone_view": tone_default,
        "drag_res": 256,
        "idle_res": 1024,
        "slat_path": current_slat_path,
        "hdri_path": current_hdri_path,
        "super_resolve": True,
    }

    # Connection-specific rendering variables (cached)
    conn_slat_path = current_slat_path
    conn_hdri_path = current_hdri_path
    conn_hs = None
    conn_rfs = None
    conn_hdri_np = None
    conn_hdri_cond = None
    conn_hdri_rot_deg = 0.0

    # Event to signal the background render worker
    render_event = asyncio.Event()

    async def render_worker():
        """Background worker that serializes and executes rendering tasks."""
        nonlocal client_state, conn_slat_path, conn_hdri_path, conn_hs, conn_rfs, conn_hdri_np, conn_hdri_cond, conn_hdri_rot_deg
        last_rendered_state = None

        # Preload default SLaT and HDRI if not in mock mode
        if not MOCK_MODE:
            try:
                async with gpu_lock:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, lambda: pipeline_manager.load_models())
                    
                    if Path(conn_slat_path).is_file():
                        slat = await loop.run_in_executor(None, lambda: pipeline_manager.pipeline.load_slat(conn_slat_path))
                        conn_hs, conn_rfs = await loop.run_in_executor(None, lambda: pipeline_manager.pipeline.decoder_pbr_feats(slat))
                        
                    if Path(conn_hdri_path).is_file():
                        conn_hdri_np = await loop.run_in_executor(None, lambda: pipeline_manager.pipeline.load_hdri(conn_hdri_path))
            except Exception as e:
                print(f"[WebSocket] Preload error: {e}")

        while True:
            await render_event.wait()
            render_event.clear()

            # Copy current state to avoid race conditions during rendering
            current_state = client_state.copy()
            if current_state == last_rendered_state:
                continue

            try:
                # 1. Handle dynamic SLaT switching
                if current_state["slat_path"] != conn_slat_path:
                    print(f"[WebSocket] Loading new SLaT: {current_state['slat_path']}", flush=True)
                    if not MOCK_MODE:
                        async with gpu_lock:
                            loop = asyncio.get_running_loop()
                            # Ensure models are loaded before loading SLaT
                            await loop.run_in_executor(None, lambda: pipeline_manager.load_models())
                            new_slat = await loop.run_in_executor(
                                None, 
                                lambda: pipeline_manager.pipeline.load_slat(current_state["slat_path"])
                            )
                            new_hs, new_rfs = await loop.run_in_executor(
                                None,
                                lambda: pipeline_manager.pipeline.decoder_pbr_feats(new_slat)
                            )
                            conn_hs = new_hs
                            conn_rfs = new_rfs
                    conn_slat_path = current_state["slat_path"]

                # 2. Handle dynamic HDRI switching
                if current_state["hdri_path"] != conn_hdri_path:
                    print(f"[WebSocket] Loading new HDRI: {current_state['hdri_path']}", flush=True)
                    if not MOCK_MODE:
                        loop = asyncio.get_running_loop()
                        new_hdri_np = await loop.run_in_executor(
                            None, 
                            lambda: pipeline_manager.pipeline.load_hdri(current_state["hdri_path"])
                        )
                        conn_hdri_np = new_hdri_np
                        conn_hdri_cond = None  # Force re-encoding
                    conn_hdri_path = current_state["hdri_path"]

                # 3. Handle HDRI rotation and encoding
                if not MOCK_MODE and (conn_hdri_cond is None or current_state["hdri_rot_deg"] != conn_hdri_rot_deg):
                    if conn_hdri_np is None:
                        if Path(conn_hdri_path).is_file():
                            print(f"[WebSocket] Loading HDRI on demand: {conn_hdri_path}", flush=True)
                            loop = asyncio.get_running_loop()
                            # Ensure models are loaded first
                            await loop.run_in_executor(None, lambda: pipeline_manager.load_models())
                            conn_hdri_np = await loop.run_in_executor(
                                None, 
                                lambda: pipeline_manager.pipeline.load_hdri(conn_hdri_path)
                            )
                    
                    if conn_hdri_np is not None:
                        async with gpu_lock:
                            loop = asyncio.get_running_loop()
                            conn_hdri_cond = await loop.run_in_executor(
                                None,
                                lambda: pipeline_manager.pipeline.encode_hdri(conn_hdri_np, current_state["hdri_rot_deg"])
                            )
                    conn_hdri_rot_deg = current_state["hdri_rot_deg"]

                # Determine render resolution
                res = current_state["drag_res"] if current_state["is_dragging"] else current_state["idle_res"]
                use_sr = current_state.get("super_resolve", True) and not MOCK_MODE
                internal_res = max(64, res // 4) if use_sr else res

                if MOCK_MODE:
                    # CPU Mock Rendering
                    from app_web.pipelines import render_mock_sphere
                    loop = asyncio.get_running_loop()
                    rgb = await loop.run_in_executor(
                        None,
                        lambda: render_mock_sphere(
                            yaw=current_state["yaw"],
                            pitch=current_state["pitch"],
                            radius=current_state["radius"],
                            hdri_rot=current_state["hdri_rot_deg"],
                            res=res,
                            tone_view=current_state["tone_view"],
                            slat_name=Path(current_state["slat_path"]).stem,
                            hdri_name=Path(current_state["hdri_path"]).stem
                        )
                    )
                else:
                    # Ensure models and SLaT are loaded if missed during startup/preload
                    if conn_hs is None or conn_rfs is None:
                        if Path(conn_slat_path).is_file():
                            print(f"[WebSocket] Loading SLaT on demand: {conn_slat_path}", flush=True)
                            async with gpu_lock:
                                loop = asyncio.get_running_loop()
                                await loop.run_in_executor(None, lambda: pipeline_manager.load_models())
                                slat = await loop.run_in_executor(None, lambda: pipeline_manager.pipeline.load_slat(conn_slat_path))
                                conn_hs, conn_rfs = await loop.run_in_executor(None, lambda: pipeline_manager.pipeline.decoder_pbr_feats(slat))
                        else:
                            raise RuntimeError(f"No valid SLaT file found at {conn_slat_path} to render.")

                    if conn_hdri_cond is None:
                        raise RuntimeError("Lighting conditions (HDRI cond) are None.")

                    # GPU Neural Rendering
                    async with gpu_lock:
                        loop = asyncio.get_running_loop()
                        
                        # Set tone mapper if changed with robust fallback mapping
                        if last_rendered_state is None or last_rendered_state["tone_view"] != current_state["tone_view"]:
                            target_view = current_state["tone_view"]
                            # Gracefully map old placeholders to standard OCIO views to prevent crashes
                            if target_view == "sRGB":
                                target_view = "Standard"
                            elif target_view == "Linear":
                                target_view = "Raw"
                            
                            try:
                                pipeline_manager.pipeline.setup_tone_mapper(target_view)
                            except Exception as e:
                                print(f"[WebSocket] Warning: failed to set tone mapper view '{target_view}': {e}. Falling back to AgX.")
                                try:
                                    pipeline_manager.pipeline.setup_tone_mapper("AgX")
                                except Exception:
                                    pass

                        rgb = await loop.run_in_executor(
                            None,
                            lambda: pipeline_manager.pipeline.render_relight_color_numpy(
                                conn_hs,
                                conn_rfs,
                                conn_hdri_cond,
                                yaw_deg=current_state["yaw"],
                                pitch_deg=current_state["pitch"],
                                fov_deg=current_state["fov"],
                                radius=current_state["radius"],
                                internal_resolution=internal_res,
                                bg_color=(0.0, 0.0, 0.0),
                                clip_near=0.05,
                                clip_far=32.0,
                            )
                        )
                        
                        if use_sr:
                            rgb = await loop.run_in_executor(
                                None,
                                lambda: pipeline_manager.super_resolve_numpy(rgb)
                            )

                # Compress the raw RGB numpy array to PNG in-memory (lossless, artifact-free)
                from app_web.pipelines import compress_to_png
                loop = asyncio.get_running_loop()
                png_bytes = await loop.run_in_executor(
                    None,
                    lambda: compress_to_png(rgb, compression_level=1)
                )

                # Send binary PNG bytes over WebSocket
                await websocket.send_bytes(png_bytes)
                last_rendered_state = current_state

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[WebSocket] Render error: {e}")
                import traceback
                traceback.print_exc()
                try:
                    await websocket.send_json({"type": "error", "message": f"Render error: {str(e)}"})
                except Exception:
                    pass

    # Start background render worker task
    worker_task = asyncio.create_task(render_worker())

    try:
        while True:
            # Receive state/event updates from the client
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "generate":
                # Handle single-image to 3D generation background task
                image_rel_path = data.get("image_path")
                seed = int(data.get("seed", 42))
                
                if not image_rel_path:
                    await websocket.send_json({"type": "error", "message": "No image path provided."})
                    continue
                    
                image_path = APP_DIR / image_rel_path
                session_dir = image_path.parent
                
                print(f"[WebSocket] Starting 3D generation for {image_path}", flush=True)
                
                loop = asyncio.get_running_loop()
                def progress_cb(progress_val: float, message: str):
                    asyncio.run_coroutine_threadsafe(
                        websocket.send_json({
                            "type": "progress",
                            "progress": progress_val,
                            "message": message
                        }),
                        loop
                    )

                try:
                    # Run the complete 3D generation pipeline task
                    slat_path, video_path, hdri_video_path = await generate_3d_asset_task(
                        image_path,
                        seed,
                        session_dir,
                        progress_cb
                    )
                    
                    # Send completion event with new assets
                    video_url = f"tmp_web_app/{session_dir.name}/{video_path.name}"
                    hdri_video_url = f"tmp_web_app/{session_dir.name}/{hdri_video_path.name}"
                    
                    # Refresh available SLaTs
                    slats = get_available_slats(str(slat_path))
                    
                    await websocket.send_json({
                        "type": "generation_complete",
                        "slat_path": str(slat_path),
                        "video_url": video_url,
                        "hdri_video_url": hdri_video_url,
                        "slats": slats
                    })
                    
                    # Automatically switch client state to the newly generated SLaT
                    client_state["slat_path"] = str(slat_path)
                    render_event.set()
                    
                except Exception as e:
                    print(f"[WebSocket] Generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({"type": "error", "message": f"Generation failed: {str(e)}"})

            elif msg_type == "generate_style_transfer":
                # Handle dual-image Style Transfer background task
                geo_image_rel = data.get("geo_image_path")
                style_image_rel = data.get("style_image_path")
                seed = int(data.get("seed", 42))
                
                if not geo_image_rel or not style_image_rel:
                    await websocket.send_json({"type": "error", "message": "Geometry image and style image are both required."})
                    continue
                    
                geo_path = APP_DIR / geo_image_rel
                style_path = APP_DIR / style_image_rel
                session_dir = geo_path.parent
                
                print(f"[WebSocket] Starting Style Transfer. Geometry: {geo_path}, Style: {style_path}", flush=True)
                
                loop = asyncio.get_running_loop()
                def progress_cb(progress_val: float, message: str):
                    asyncio.run_coroutine_threadsafe(
                        websocket.send_json({
                            "type": "progress",
                            "progress": progress_val,
                            "message": message
                        }),
                        loop
                    )

                try:
                    # Run the complete Style Transfer 3D generation pipeline task
                    slat_path, video_path, hdri_video_path = await generate_style_transfer_task(
                        geo_path,
                        style_path,
                        seed,
                        session_dir,
                        progress_cb
                    )
                    
                    # Send completion event with new assets
                    video_url = f"tmp_web_app/{session_dir.name}/{video_path.name}"
                    hdri_video_url = f"tmp_web_app/{session_dir.name}/{hdri_video_path.name}"
                    
                    # Refresh available SLaTs
                    slats = get_available_slats(str(slat_path))
                    
                    await websocket.send_json({
                        "type": "generation_complete",
                        "slat_path": str(slat_path),
                        "video_url": video_url,
                        "hdri_video_url": hdri_video_url,
                        "slats": slats
                    })
                    
                    # Automatically switch client state to the newly generated SLaT
                    client_state["slat_path"] = str(slat_path)
                    render_event.set()
                    
                except Exception as e:
                    print(f"[WebSocket] Style Transfer failed: {e}")
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({"type": "error", "message": f"Style Transfer failed: {str(e)}"})
                
            elif msg_type == "export_glb":
                # Handle baking PBR textures and exporting GLB mesh
                slat_path = data.get("slat_path", client_state["slat_path"])
                hdri_path = data.get("hdri_path", client_state["hdri_path"])
                simplify = float(data.get("simplify", 0.9))
                texture_size = int(data.get("texture_size", 1024))
                
                session_dir = Path(slat_path).parent
                if session_dir == APP_DIR / "assets/example_slats":
                    # If using default, save exported GLB in a generic tmp session
                    session_id = str(uuid.uuid4())[:8]
                    session_dir = CACHE_DIR / session_id
                    session_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"[WebSocket] Exporting GLB from {slat_path}...", flush=True)
                
                loop = asyncio.get_running_loop()
                def progress_cb(progress_val: float, message: str):
                    asyncio.run_coroutine_threadsafe(
                        websocket.send_json({
                            "type": "progress",
                            "progress": progress_val,
                            "message": message
                        }),
                        loop
                    )
                    
                try:
                    glb_path = await export_glb_task(
                        Path(slat_path),
                        Path(hdri_path),
                        simplify,
                        texture_size,
                        session_dir,
                        progress_cb
                    )
                    
                    glb_url = f"tmp_web_app/{session_dir.name}/{glb_path.name}"
                    await websocket.send_json({
                        "type": "glb_complete",
                        "glb_url": glb_url,
                        "glb_filename": glb_path.name
                    })
                except Exception as e:
                    print(f"[WebSocket] GLB Export failed: {e}")
                    await websocket.send_json({"type": "error", "message": f"GLB Export failed: {str(e)}"})
                    
            else:
                # Standard render/camera state update
                client_state.update({
                    "yaw": float(data.get("yaw", client_state["yaw"])),
                    "pitch": float(data.get("pitch", client_state["pitch"])),
                    "radius": float(data.get("radius", client_state["radius"])),
                    "fov": float(data.get("fov", client_state["fov"])),
                    "is_dragging": bool(data.get("is_dragging", client_state["is_dragging"])),
                    "hdri_rot_deg": float(data.get("hdri_rot_deg", client_state["hdri_rot_deg"])),
                    "tone_view": str(data.get("tone_view", client_state["tone_view"])),
                    "drag_res": int(data.get("drag_res", client_state["drag_res"])),
                    "idle_res": int(data.get("idle_res", client_state["idle_res"])),
                    "slat_path": str(data.get("slat_path", client_state["slat_path"])),
                    "hdri_path": str(data.get("hdri_path", client_state["hdri_path"])),
                    "super_resolve": bool(data.get("super_resolve", client_state["super_resolve"])),
                })

                # Trigger render worker
                render_event.set()

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    finally:
        # Clean up render worker task
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
