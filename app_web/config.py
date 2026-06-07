import argparse
import os
from pathlib import Path
import torch

APP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_HDRI = APP_DIR / "assets/hdris/studio_small_03_1k.exr"
DEFAULT_SLAT = APP_DIR / (
    "assets/example_slats/"
    "2a0d671ce308adb93323eae7141953fc1a5ba68f38cc69f476d5e904c634864d.npz"
)

# Mock Mode configuration (for CPU-only environments)
MOCK_MODE = not torch.cuda.is_available()

# Temp directory for uploaded images and session outputs
CACHE_DIR = APP_DIR / "tmp_web_app"

# Clean up previous cache on server startup to free disk space and avoid cluttering SLaT selection
if CACHE_DIR.exists():
    import shutil
    try:
        shutil.rmtree(CACHE_DIR)
        print(f"[Startup] Cleaned up previous cache directory: {CACHE_DIR}", flush=True)
    except Exception as e:
        print(f"[Startup] Warning: Failed to clear cache directory: {e}", flush=True)

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Shared global arguments
args = None

def parse_args() -> argparse.Namespace:
    global args
    p = argparse.ArgumentParser(description="NeAR neural relight web viewer (FastAPI)")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--slat",
        type=str,
        default=str(DEFAULT_SLAT) if DEFAULT_SLAT.exists() else "",
        help="Path to SLaT .npz (feats, coords)",
    )
    p.add_argument(
        "--hdri",
        type=str,
        default=str(DEFAULT_HDRI) if DEFAULT_HDRI.exists() else "",
        help="HDRI .exr path",
    )
    p.add_argument(
        "--pretrained",
        type=str,
        default=os.environ.get("NEAR_PRETRAINED", "luh0502/NeAR"),
    )
    p.add_argument(
        "--clip-near",
        type=float,
        default=0.05,
        help="Neural renderer near clip (view-space)",
    )
    p.add_argument(
        "--clip-far",
        type=float,
        default=32.0,
        help="Neural renderer far clip (view-space)",
    )
    p.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link using Gradio's tunnel utility",
    )
    p.add_argument(
        "--username",
        type=str,
        default=None,
        help="Username for Basic authentication to secure the app",
    )
    p.add_argument(
        "--password",
        type=str,
        default=None,
        help="Password for Basic authentication to secure the app",
    )
    args = p.parse_args()
    return args
