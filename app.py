import os
import sys
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Point Gradio's per-session cache at a stable in-repo dir. Set BEFORE
# `import gradio` so gradio's internal utils pick it up at module load.
# Gradio 6.0 dropped `Blocks.launch(tmp_dir=...)`; `GRADIO_TEMP_DIR` is the
# canonical replacement.
APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "tmp_gradio"
# Wipe the previous run's session files so the tree doesn't grow unboundedly
# across restarts. Active-session pruning is handled by gradio's `delete_cache`
# on the Blocks below.
if CACHE_DIR.exists():
    _n = sum(1 for _ in CACHE_DIR.rglob("*") if _.is_file())
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    print(f"[startup] purged {_n} stale file(s) from {CACHE_DIR}")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(CACHE_DIR))

import gradio as gr
import imageio
import numpy as np
import torch
import trimesh
from PIL import Image

try:
    import gradio_client.utils as client_utils

    _get_type_orig = client_utils.get_type

    def _get_type_patched(schema):
        if isinstance(schema, bool):
            return "boolean"
        return _get_type_orig(schema)

    client_utils.get_type = _get_type_patched
except Exception:
    pass

sys.path.insert(0, "./hy3dshape")
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0") # H series GPUs require this to avoid some cutlass errors

from trellis.pipelines import NeARImageToRelightable3DPipeline
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # pyright: ignore[reportMissingImports]


DEFAULT_IMAGE = APP_DIR / "assets/example_image/T.png"
DEFAULT_SLAT = APP_DIR / "assets/example_slats/2a0d671ce308adb93323eae7141953fc1a5ba68f38cc69f476d5e904c634864d.npz"
DEFAULT_HDRI = APP_DIR / "assets/hdris/studio_small_03_1k.exr"
DEFAULT_PORT = 7812
MAX_SEED = np.iinfo(np.int32).max


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def ensure_session_dir(req: Optional[gr.Request]) -> Path:
    session_id = getattr(req, "session_hash", None) or "shared"
    d = CACHE_DIR / str(session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def clear_session_dir(req: Optional[gr.Request]) -> str:
    d = ensure_session_dir(req)
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    return "Session cache cleared."


def end_session(req: gr.Request):
    d = ensure_session_dir(req)
    shutil.rmtree(d, ignore_errors=True)


def get_file_path(file_obj: Any) -> Optional[str]:
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    for attr in ("name", "path", "value"):
        v = getattr(file_obj, attr, None)
        if isinstance(v, str) and v:
            return v
    return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GEOMETRY_PIPELINE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
GEOMETRY_PIPELINE.to(DEVICE)

PIPELINE = NeARImageToRelightable3DPipeline.from_pretrained(os.environ.get("NEAR_PRETRAINED", "luh0502/NeAR"))
PIPELINE.to(DEVICE)

AVAILABLE_TONE_MAPPERS = getattr(PIPELINE.tone_mapper, "available_views", ["AgX"])


def set_tone_mapper(view_name: str):
    if view_name:
        PIPELINE.setup_tone_mapper(view_name)


def preview_hdri(hdri_file_obj: Any, tone_mapper_name: str):
    hdri_path = get_file_path(hdri_file_obj)
    if not hdri_path:
        return None, "Upload an HDRI `.exr` (left column)."
    set_tone_mapper(tone_mapper_name)
    hdri_np = PIPELINE.load_hdri(hdri_path)
    preview = PIPELINE.tone_mapper.hdr_to_ldr(hdri_np)
    preview = (np.clip(preview, 0, 1) * 255).astype(np.uint8)
    name = Path(hdri_path).name
    return preview, f"HDRI **{name}** — preview updated."


def switch_asset_source(mode: str):
    """
    Switching mode is a fresh start: reset asset_state, clear the SLaT/image
    inputs, and blank out stale render outputs. Otherwise the previous mode's
    asset_state.slat_path leaks across, and the user sees old videos that
    look like they came from the new SLaT.
    """
    is_existing = mode == "From Existing SLaT"
    return (
        gr.Tabs(selected=1 if is_existing else 0),
        {},     # asset_state
        None,   # slat_upload
        "",     # slat_path_text
        None,   # mesh_viewer
        None,   # pbr_viewer
        None,   # color_output
        None,   # base_color_output
        None,   # metallic_output
        None,   # roughness_output
        None,   # shadow_output
        None,   # camera_video_output
        None,   # hdri_roll_video_output
        None,   # hdri_render_video_output
        "Switched mode — state reset. Re-run the workflow.",  # status_md
    )


def _ensure_rgba(img: Image.Image) -> Image.Image:
    """Normalize to RGBA so alpha is preserved for mesh (white matte) vs SLaT (black matte)."""
    if img.mode == "RGBA":
        return img
    if img.mode == "RGB":
        r, g, b = img.split()
        a = Image.new("L", img.size, 255)
        return Image.merge("RGBA", (r, g, b, a))
    return img.convert("RGBA")


@torch.inference_mode()
def preprocess_image_only(image_input: Optional[Image.Image]):
    if image_input is None:
        return None
    return PIPELINE.preprocess_image_rgba(_ensure_rgba(image_input))


def preprocess_default_image() -> Optional[Image.Image]:
    """Run once on page load so the default example is background-removed (no .change loop)."""
    img = Image.open(DEFAULT_IMAGE).convert("RGBA")
    return preprocess_image_only(img)


def save_slat_npz(slat, save_path: Path):
    np.savez(
        save_path,
        feats=slat.feats.detach().cpu().numpy(),
        coords=slat.coords.detach().cpu().numpy(),
    )


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_mesh(
    image_input: Optional[Image.Image],
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
):
    """Step ①: generate Hunyuan3D geometry from an already preprocessed image.
    Returns: (state, mesh_glb_path, status)
    """
    session_dir = ensure_session_dir(req)

    if image_input is None:
        raise gr.Error("Please upload an input image.")

    rgba = _ensure_rgba(image_input)
    if rgba.size != (518, 518):
        rgba = PIPELINE.preprocess_image_rgba(rgba)
    # Hunyuan3D mesh: composite onto white. SLaT step uses black matte separately.
    mesh_rgb = PIPELINE.flatten_rgba_on_matte(rgba, (1.0, 1.0, 1.0))
    rgba.save(session_dir / "input_preprocessed_rgba.png")
    mesh_rgb.save(session_dir / "input_processed.png")

    progress(0.6, desc="Generating geometry")
    mesh = GEOMETRY_PIPELINE(image=mesh_rgb)[0]
    mesh_path = session_dir / "initial_3d_shape.glb"
    mesh.export(mesh_path)

    state = {
        "mode": "image",
        "mesh_path": str(mesh_path),
        "processed_image_path": str(session_dir / "input_processed.png"),
        "slat_path": None,
    }
    return (
        state,
        str(mesh_path),
        "**Mesh ready** — Click **② Generate / Load SLaT** to continue.",
    )


@torch.inference_mode()
def generate_slat(
    asset_state: Dict[str, Any],
    image_input: Optional[Image.Image],
    seed: int,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
):
    session_dir = ensure_session_dir(req)

    if not asset_state or not asset_state.get("mesh_path"):
        raise gr.Error("Please run ① Generate Mesh first.")
    mesh_path = asset_state["mesh_path"]
    if not os.path.exists(mesh_path):
        raise gr.Error("Mesh file not found — please regenerate the mesh.")

    if image_input is None:
        raise gr.Error("Preprocessed image not found — please upload the image again.")

    progress(0.1, desc="Loading mesh")
    mesh = trimesh.load(mesh_path, force="mesh")
    rgba = _ensure_rgba(image_input)
    if rgba.size != (518, 518):
        rgba = PIPELINE.preprocess_image_rgba(rgba)
    slat_rgb = PIPELINE.flatten_rgba_on_matte(rgba, (0.0, 0.0, 0.0))

    progress(0.3, desc="Computing SLaT coordinates")
    coords = PIPELINE.shape_to_coords(mesh)

    progress(0.6, desc="Generating SLaT")
    slat = PIPELINE.run_with_coords([slat_rgb], coords, seed=int(seed), preprocess_image=False)

    slat_path = session_dir / "generated_slat.npz"
    save_slat_npz(slat, slat_path)

    new_state = {**asset_state, "slat_path": str(slat_path)}
    return new_state, f"**Asset ready** — SLaT generated (seed `{seed}`)."


def load_slat_file(slat_upload: Any, slat_path_text: str, req: gr.Request):
    # Prefer the text path: `gr.Examples` populates this field, but the
    # Upload widget can hold a stale file from an earlier interaction.
    text = (slat_path_text or "").strip()
    resolved = text or get_file_path(slat_upload) or ""
    if not resolved:
        raise gr.Error("Please provide a SLaT `.npz` path or upload one.")
    if not os.path.exists(resolved):
        raise gr.Error(f"SLaT file not found: `{resolved}`")
    state = {"mode": "slat", "slat_path": resolved, "mesh_path": None, "processed_image_path": None}
    return state, f"SLaT **{Path(resolved).name}** loaded."


def prepare_slat(
    source_mode: str,
    asset_state: Dict[str, Any],
    image_input: Optional[Image.Image],
    seed: int,
    slat_upload: Any,
    slat_path_text: str,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
):
    if source_mode == "From Image":
        return generate_slat(asset_state, image_input, seed, req, progress)
    return load_slat_file(slat_upload, slat_path_text, req)


def require_asset_state(asset_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not asset_state or not asset_state.get("slat_path"):
        raise gr.Error("Please generate or load a SLaT first.")
    return asset_state


def load_asset_and_hdri(asset_state: Dict[str, Any], hdri_file_obj: Any, tone_mapper_name: str):
    asset_state = require_asset_state(asset_state)
    hdri_path = get_file_path(hdri_file_obj)
    if not hdri_path:
        raise gr.Error("Please upload an HDRI `.exr` file.")
    set_tone_mapper(tone_mapper_name)
    slat = PIPELINE.load_slat(asset_state["slat_path"])
    hdri_np = PIPELINE.load_hdri(hdri_path)
    return slat, hdri_np


@torch.inference_mode()
def render_preview(
    asset_state: Dict[str, Any],
    hdri_file_obj: Any,
    tone_mapper_name: str,
    hdri_rot: float,
    yaw: float,
    pitch: float,
    fov: float,
    radius: float,
    resolution: int,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
):
    session_dir = ensure_session_dir(req)
    progress(0.1, desc="Loading SLaT and HDRI")
    slat, hdri_np = load_asset_and_hdri(asset_state, hdri_file_obj, tone_mapper_name)

    progress(0.5, desc="Rendering")
    views = PIPELINE.render_view(
        slat, hdri_np,
        yaw_deg=yaw, pitch_deg=pitch, fov=fov, radius=radius,
        hdri_rot_deg=hdri_rot, resolution=int(resolution),
    )
    for key, image in views.items():
        image.save(session_dir / f"preview_{key}.png")

    msg = (
        f"**Preview done** — "
        f"yaw `{yaw:.0f}°` pitch `{pitch:.0f}°` · "
        f"fov `{fov:.0f}` radius `{radius:.1f}` · HDRI rot `{hdri_rot:.0f}°`"
    )
    return (
        views["color"],
        views["base_color"],
        views["metallic"],
        views["roughness"],
        views["shadow"],
        msg,
    )


@torch.inference_mode()
def render_camera_video(
    asset_state: Dict[str, Any],
    hdri_file_obj: Any,
    tone_mapper_name: str,
    hdri_rot: float,
    fps: int,
    num_views: int,
    fov: float,
    radius: float,
    full_video: bool,
    shadow_video: bool,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
):
    session_dir = ensure_session_dir(req)
    progress(0.1, desc="Loading SLaT and HDRI")
    slat, hdri_np = load_asset_and_hdri(asset_state, hdri_file_obj, tone_mapper_name)

    progress(0.4, desc="Rendering camera path")
    frames = PIPELINE.render_camera_path_video(
        slat, hdri_np,
        num_views=int(num_views), fov=fov, radius=radius,
        hdri_rot_deg=hdri_rot, full_video=full_video, shadow_video=shadow_video,
        bg_color=(1, 1, 1), verbose=True,
    )
    # Unique filename per call: gradio's gr.Video doesn't refresh when the
    # returned path string is identical to the previous one, even if the
    # underlying file content changed (browser caches by URL).
    stamp = time.time_ns()
    kind = "camera_path_full" if full_video else "camera_path"
    video_path = session_dir / f"{kind}_{stamp}.mp4"
    imageio.mimsave(video_path, frames, fps=int(fps))
    return str(video_path), f"**Camera path video saved**"


@torch.inference_mode()
def render_hdri_video(
    asset_state: Dict[str, Any],
    hdri_file_obj: Any,
    tone_mapper_name: str,
    fps: int,
    num_frames: int,
    yaw: float,
    pitch: float,
    fov: float,
    radius: float,
    full_video: bool,
    shadow_video: bool,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
):
    session_dir = ensure_session_dir(req)
    progress(0.1, desc="Loading SLaT and HDRI")
    slat, hdri_np = load_asset_and_hdri(asset_state, hdri_file_obj, tone_mapper_name)

    progress(0.4, desc="Rendering HDRI rotation")
    hdri_roll_frames, render_frames = PIPELINE.render_hdri_rotation_video(
        slat, hdri_np,
        num_frames=int(num_frames), yaw_deg=yaw, pitch_deg=pitch,
        fov=fov, radius=radius, full_video=full_video, shadow_video=shadow_video,
        bg_color=(1, 1, 1), verbose=True,
    )
    stamp = time.time_ns()
    hdri_roll_path = session_dir / f"hdri_roll_{stamp}.mp4"
    render_kind = "hdri_rotation_full" if full_video else "hdri_rotation"
    render_path = session_dir / f"{render_kind}_{stamp}.mp4"
    imageio.mimsave(hdri_roll_path, hdri_roll_frames, fps=int(fps))
    imageio.mimsave(render_path, render_frames, fps=int(fps))
    return str(hdri_roll_path), str(render_path), "**HDRI rotation video saved**"


def export_glb(
    asset_state: Dict[str, Any],
    hdri_file_obj: Any,
    tone_mapper_name: str,
    hdri_rot: float,
    simplify: float,
    texture_size: int,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
):
    """Returns: (glb_path, status)"""
    session_dir = ensure_session_dir(req)
    progress(0.1, desc="Loading SLaT and HDRI")
    slat, hdri_np = load_asset_and_hdri(asset_state, hdri_file_obj, tone_mapper_name)

    progress(0.6, desc="Baking PBR textures")
    glb = PIPELINE.export_glb_from_slat(
        slat, hdri_np,
        hdri_rot_deg=hdri_rot, base_mesh=None,
        simplify=simplify, texture_size=int(texture_size), fill_holes=True,
    )
    glb_path = session_dir / f"near_pbr_{time.time_ns()}.glb"
    glb.export(glb_path)
    return str(glb_path), f"PBR GLB exported: **{glb_path.name}**"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* Use full browser width (was max-width:1600px leaving empty margin on the right) */
.gradio-container { max-width: 100% !important; width: 100% !important; }
main.gradio-container { max-width: 100% !important; }
.gradio-wrap { max-width: 100% !important; }

/* Top header: TRELLIS-style left-aligned title + bullets */
.near-app-header {
  text-align: left !important;
  padding: 0.35rem 0 1.1rem 0 !important;
  margin: 0 !important;
}
.near-app-header .prose,
.near-app-header p { margin: 0 !important; }
.near-app-header h2 {
  font-size: clamp(1.35rem, 2.4vw, 1.85rem) !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em !important;
  margin: 0 0 0.45rem 0 !important;
  line-height: 1.25 !important;
}
.near-app-header h2 a {
  color: var(--link-text-color, var(--color-accent)) !important;
  text-decoration: none !important;
}
.near-app-header h2 a:hover { text-decoration: underline !important; }
.near-app-header ul {
  margin: 0 !important;
  padding-left: 1.2rem !important;
  font-size: 0.88rem !important;
  color: #4b5563 !important;
  line-height: 1.45 !important;
}
.near-app-header li { margin: 0.15rem 0 !important; }

/* Left column: compact section labels (no numbered circles) */
.section-kicker {
  font-size: 0.7rem !important;
  font-weight: 700 !important;
  color: #9ca3af !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  margin: 0 0 0.45rem 0 !important;
  padding: 0 !important;
}

/* HDRI file picker: light card instead of default dark block */
.hdri-upload-zone,
.hdri-file-input,
.hdri-upload-zone .upload-container,
.hdri-upload-zone [data-testid="file-upload"],
.hdri-file-input [data-testid="file-upload"],
.hdri-upload-zone .file-preview,
.hdri-file-input .file-preview,
.hdri-upload-zone .wrap,
.hdri-file-input .wrap,
.hdri-upload-zone .panel,
.hdri-file-input .panel {
  background: #f9fafb !important;
  border-color: #e5e7eb !important;
  color: #374151 !important;
}
.hdri-upload-zone .file-preview,
.hdri-file-input .file-preview { border-radius: 8px !important; }
.hdri-upload-zone .label-wrap,
.hdri-file-input .label-wrap { color: #4b5563 !important; }

/* HDRI preview image: remove thick / black frame (Gradio panel border) */
.hdri-preview-image,
.hdri-preview-image.panel,
.hdri-preview-image .wrap,
.hdri-preview-image .image-container,
.hdri-preview-image .image-frame,
.hdri-preview-image .image-wrapper,
.hdri-preview-image [data-testid="image"],
.hdri-preview-image .icon-buttons,
.hdri-preview-image img {
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
}
.hdri-preview-image img {
  border-radius: 8px !important;
}

/* Export accordion: remove heavy black box; keep a light separator on the header only */
.export-accordion,
.export-accordion.panel,
.export-accordion > div,
.export-accordion details,
.export-accordion .label-wrap,
.export-accordion .accordion-header {
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
}
.export-accordion summary,
.export-accordion .label-wrap {
  border-bottom: 1px solid #e5e7eb !important;
  background: transparent !important;
}

/* Gradio 4+ block chrome sometimes forces --block-border-color */
.gradio-container .hdri-preview-image,
.gradio-container .export-accordion {
  --block-border-width: 0px !important;
  --panel-border-width: 0 !important;
}

/* Shadow map preview: same flat frame as HDRI preview */
.shadow-preview-image,
.shadow-preview-image.panel,
.shadow-preview-image .wrap,
.shadow-preview-image .image-container,
.shadow-preview-image .image-frame,
.shadow-preview-image .image-wrapper,
.shadow-preview-image [data-testid="image"],
.shadow-preview-image img {
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
}
.shadow-preview-image img { border-radius: 8px !important; }
.gradio-container .shadow-preview-image {
  --block-border-width: 0px !important;
  --panel-border-width: 0 !important;
}

/* Main output tabs: larger, easier to spot */
.main-output-tabs > .tab-nav,
.main-output-tabs .tab-nav button {
  font-size: 0.95rem !important;
  font-weight: 600 !important;
}
.main-output-tabs .tab-nav button { padding: 0.45rem 0.9rem !important; }

/* Status strip: one left accent only (Gradio panel also draws accent — disable it here) */
.gradio-container .status-footer,
.status-footer.panel,
.status-footer.block {
  --block-border-width: 0px !important;
  --panel-border-width: 0px !important;
}
.status-footer {
  font-size: 0.8125rem !important;
  line-height: 1.45 !important;
  color: var(--body-text-color-subdued, #6b7280) !important;
  margin: 0 0 0.65rem 0 !important;
  padding: 0.5rem 0.65rem 0.5rem 0.7rem !important;
  background: var(--block-background-fill, #f9fafb) !important;
  /* Single box: one thick left edge (avoid stacking with Gradio .block border) */
  border-width: 1px 1px 1px 3px !important;
  border-style: solid !important;
  border-color: var(--border-color-primary, #e5e7eb) var(--border-color-primary, #e5e7eb)
    var(--border-color-primary, #e5e7eb) var(--color-accent, #2563eb) !important;
  border-radius: 8px !important;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05) !important;
}
.status-footer .form,
.status-footer .wrap,
.status-footer .prose,
.status-footer .prose > *:first-child {
  border: none !important;
  box-shadow: none !important;
}
.status-footer .prose blockquote {
  border-left: none !important;
  padding-left: 0 !important;
  margin-left: 0 !important;
}
.status-footer p,
.status-footer .prose p {
  margin: 0 !important;
  line-height: 1.05 !important;
}
.status-footer strong {
  color: var(--body-text-color, #374151) !important;
  font-weight: 600 !important;
}
.status-footer a {
  color: var(--link-text-color, var(--color-accent, #2563eb)) !important;
  text-decoration: none !important;
}
.status-footer a:hover { text-decoration: underline !important; }

.ctrl-strip {
  border:1px solid #e5e7eb; border-radius:8px;
  padding:0.55rem 0.8rem 0.4rem; margin-bottom:0.6rem; background:#fff;
}
.ctrl-strip-title {
  font-size:0.72rem; font-weight:600; color:#9ca3af;
  text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;
}

.mat-label {
  font-size:0.72rem; font-weight:700; color:#9ca3af;
  text-transform:uppercase; letter-spacing:0.07em; margin:0.7rem 0 0.2rem;
}

.divider { border:none; border-top:1px solid #e5e7eb; margin:0.5rem 0; }

.img-gallery table { display:grid !important; grid-template-columns:repeat(3,1fr) !important; gap:3px !important; }
.img-gallery table thead { display:none !important; }
.img-gallery table tr { display:contents !important; }
.img-gallery table td { padding:0 !important; }
.img-gallery table td img { width:100% !important; height:68px !important; object-fit:cover !important; border-radius:5px !important; }

.hdri-gallery table { display:grid !important; grid-template-columns:repeat(2,1fr) !important; gap:3px !important; }
.hdri-gallery table thead { display:none !important; }
.hdri-gallery table tr { display:contents !important; }
.hdri-gallery table td { padding:0 !important; font-size:0.76rem; text-align:center; word-break:break-all; }

/* Right sidebar: align with TRELLIS-style narrow examples column */
.sidebar-examples { min-width: 0 !important; }
.sidebar-examples .label-wrap { font-size: 0.85rem !important; }
.gradio-container .sidebar-examples table { width: 100% !important; }

footer { display:none !important; }
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    # Gradio 6.0 moved `theme` and `css` out of the Blocks constructor — they
    # are passed to `launch()` further down instead.
    #
    # `delete_cache=(sweep_every, max_age)`: every `sweep_every` seconds,
    # delete files older than `max_age`. The previous (600, 600) deleted
    # uploaded images after 10 min — long-running interactions then hit
    # FileNotFoundError when the UI re-referenced the purged path. Widen
    # to (1800, 3600) so files survive a typical session.
    with gr.Blocks(
        title="NeAR",
        delete_cache=(1800, 3600),
        fill_width=True,
    ) as demo:
        asset_state = gr.State({})

        gr.Markdown(
            """
## Single Image to Relightable 3DGS with [NeAR](https://near-project.github.io/)
* Upload an RGBA image (or load an existing SLaT), run **Generate Mesh** then **Generate / Load SLaT**, pick an HDRI, and use **Camera & HDRI** to relight.
* Use **Geometry** for mesh / PBR preview, **Preview** for still renders, **Videos** for camera or HDRI paths; **Export PBR GLB** when you are happy with the result.
* Texture style transfer is possible when the reference images used for **mesh** and **SLaT** are different.
            """,
            elem_classes=["near-app-header"],
        )

        _img_ex = [[str(p)] for p in sorted((APP_DIR / "assets/example_image").glob("*.png"))]
        _slat_ex = [[str(p)] for p in sorted((APP_DIR / "assets/example_slats").glob("*.npz"))]
        _hdri_ex = [[str(p)] for p in sorted((APP_DIR / "assets/hdris").glob("*.exr"))]

        with gr.Row(equal_height=False):

            # ════════════════════════════════════════════════════════════════
            # LEFT — controls only (TRELLIS-style narrow column)
            # ════════════════════════════════════════════════════════════════
            with gr.Column(scale=1, min_width=360):

                with gr.Group():
                    gr.HTML('<p class="section-kicker">Asset</p>')
                    source_mode = gr.Radio(
                        ["From Image", "From Existing SLaT"],
                        value="From Image",
                        label="",
                        show_label=False,
                    )
                    with gr.Tabs(selected=0) as source_tabs:

                        with gr.Tab("Image", id=0):
                            image_input = gr.Image(
                                label="Input Image", type="pil", image_mode="RGBA",
                                value=str(DEFAULT_IMAGE) if DEFAULT_IMAGE.exists() else None,
                                height=400,
                            )
                            seed = gr.Slider(0, MAX_SEED, value=42, step=1, label="Seed (SLaT)")
                            mesh_button = gr.Button("① Generate Mesh", variant="primary", min_width=100)

                        with gr.Tab("SLaT", id=1):
                            slat_upload = gr.File(label="Upload SLaT (.npz)", file_types=[".npz"])
                            slat_path_text = gr.Textbox(
                                label="Or enter local path",
                                placeholder="/path/to/sample_slat.npz",
                            )

                    slat_button = gr.Button(
                        "② Generate / Load SLaT", variant="primary", min_width=100,
                    )
                    # gr.HTML(
                        # "<div style='font-size:0.78rem;color:#9ca3af;margin-top:0.2rem;'>"
                        # "Image mode: run ① then ②. SLaT mode: ② loads file directly.</div>"
                    # )

                with gr.Group():
                    gr.HTML('<p class="section-kicker">HDRI</p>')
                    with gr.Column(elem_classes=["hdri-upload-zone"]):
                        hdri_file = gr.File(
                            label="Environment (.exr)", file_types=[".exr"],
                            value=str(DEFAULT_HDRI) if DEFAULT_HDRI.exists() else None,
                            elem_classes=["hdri-file-input"],
                        )
                    hdri_preview = gr.Image(
                        label="Preview",
                        interactive=False,
                        height=130,
                        container=False,
                        elem_classes=["hdri-preview-image"],
                    )

                with gr.Group():
                    gr.HTML('<p class="section-kicker">Export</p>')
                    with gr.Accordion(
                        "Export Settings",
                        open=False,
                        elem_classes=["export-accordion"],
                    ):
                        with gr.Row():
                            simplify     = gr.Slider(0.8, 0.99, value=0.95, step=0.01, label="Mesh Simplify")
                            texture_size = gr.Slider(512, 4096, value=2048, step=512,  label="Texture Size")

                with gr.Row():
                    clear_button = gr.Button("Clear Cache", variant="secondary", min_width=100)

            # ════════════════════════════════════════════════════════════════
            # CENTER — status at top, then Camera & HDRI, then tabs
            # ════════════════════════════════════════════════════════════════
            with gr.Column(scale=10, min_width=560):

                status_md = gr.Markdown(
                    "Ready — use **Asset** (left) and **HDRI** to begin.",
                    elem_classes=["status-footer"],
                )


                with gr.Group(elem_classes=["ctrl-strip"]):
                    gr.HTML("<div class='ctrl-strip-title'>Camera &amp; HDRI</div>")
                    with gr.Row():
                        tone_mapper_name = gr.Dropdown(
                            choices=AVAILABLE_TONE_MAPPERS,
                            value=AVAILABLE_TONE_MAPPERS[0] if AVAILABLE_TONE_MAPPERS else None,
                            label="Tone Mapper", min_width=120,
                        )
                        hdri_rot   = gr.Slider(0, 360,   value=0,   step=1,   label="HDRI Rotation °")
                        resolution = gr.Slider(256, 1024, value=512, step=256, label="Preview Res")
                    with gr.Row():
                        yaw    = gr.Slider(0, 360,   value=0,   step=0.5,  label="Yaw °")
                        pitch  = gr.Slider(-90, 90,  value=0,   step=0.5,  label="Pitch °")
                        fov    = gr.Slider(10, 70,   value=40,  step=1,    label="FoV")
                        radius = gr.Slider(1.0, 4.0, value=2.0, step=0.05, label="Radius")

                with gr.Tabs(elem_classes=["main-output-tabs"]):

                    with gr.Tab("Geometry", id=0):
                        with gr.Row():
                            mesh_viewer = gr.Model3D(
                                label="3D Mesh", interactive=False, height=520,
                            )
                            pbr_viewer = gr.Model3D(
                                label="PBR GLB", interactive=False, height=520,
                            )
                        gr.HTML("<hr class='divider'>")
                        with gr.Row():
                            export_glb_button = gr.Button("Export PBR GLB", variant="primary", min_width=140)

                    with gr.Tab("Preview", id=1):
                        gr.HTML(
                            "<p style='font-size:0.78rem;color:#9ca3af;margin:0 0 0.35rem 0;'>"
                            "Use <b>Camera &amp; HDRI</b> under the tabs, then render.</p>"
                        )
                        preview_button = gr.Button("Render Preview", variant="primary", min_width=100)
                        gr.HTML("<hr class='divider'>")
                        with gr.Row():
                            color_output = gr.Image(label="Relit Result", interactive=False, height=400)
                            with gr.Column():
                                with gr.Row():
                                    base_color_output = gr.Image(label="Base Color", interactive=False, height=200)
                                    metallic_output   = gr.Image(label="Metallic",   interactive=False, height=200)
                                with gr.Row():
                                    roughness_output  = gr.Image(label="Roughness",  interactive=False, height=200)
                                    shadow_output = gr.Image(label="Shadow", interactive=False, height=200)

                    with gr.Tab("Videos", id=2):
                        with gr.Accordion("Video Settings", open=False):
                            with gr.Row():
                                fps        = gr.Slider(1, 60,  value=24, step=1, label="FPS")
                                num_views  = gr.Slider(8, 120, value=40, step=1, label="Camera Frames")
                                num_frames = gr.Slider(8, 120, value=40, step=1, label="HDRI Frames")
                            with gr.Row():
                                full_video = gr.Checkbox(label="Full composite video", value=True)
                                shadow_video = gr.Checkbox(
                                    label="Include shadow in video",
                                    value=True,
                                )
                        with gr.Row():
                            camera_video_button = gr.Button("Camera Path Video",   variant="primary",   min_width=100)
                            hdri_video_button   = gr.Button("HDRI Rotation Video", variant="primary",   min_width=100)
                        camera_video_output = gr.Video(
                            label="Camera Path", autoplay=True, loop=True, height=340,
                        )
                        hdri_render_video_output = gr.Video(
                            label="HDRI Rotation Render", autoplay=True, loop=True, height=300,
                        )
                        with gr.Accordion("HDRI Roll (environment panorama)", open=False):
                            hdri_roll_video_output = gr.Video(
                                label="HDRI Roll", autoplay=True, loop=True, height=180,
                            )


            # ════════════════════════════════════════════════════════════════
            # RIGHT — examples sidebar (TRELLIS-style narrow column)
            # ════════════════════════════════════════════════════════════════
            with gr.Column(scale=1, min_width=172):
                with gr.Column(visible=True, elem_classes=["sidebar-examples", "img-gallery"]) as col_img_examples:
                    if _img_ex:
                        gr.Examples(
                            examples=_img_ex,
                            inputs=[image_input],
                            fn=preprocess_image_only,
                            outputs=[image_input],
                            run_on_click=True,
                            examples_per_page=18,
                            label="Examples",
                        )
                    else:
                        gr.Markdown("*No PNG examples in `assets/example_image`*")

                with gr.Column(visible=False, elem_classes=["sidebar-examples"]) as col_slat_examples:
                    if _slat_ex:
                        gr.Examples(
                            examples=_slat_ex,
                            inputs=[slat_path_text],
                            label="Example SLaTs",
                        )
                    else:
                        gr.Markdown("*No `.npz` examples in `assets/example_slats`*")

                with gr.Column(visible=True, elem_classes=["sidebar-examples", "hdri-gallery"]) as col_hdri_examples:
                    if _hdri_ex:
                        gr.Examples(
                            examples=_hdri_ex,
                            inputs=[hdri_file],
                            label="Example HDRIs",
                            examples_per_page=8,
                        )
                    else:
                        gr.Markdown("*No `.exr` examples in `assets/hdris`*")

        # ── Event wiring ─────────────────────────────────────────────────────
        demo.unload(end_session)

        # Default image: preprocess once on load. Do NOT use image_input.change → outputs=[image_input]:
        # that retriggers change forever (spinner) because updating the same component fires change again.
        if DEFAULT_IMAGE.exists():
            demo.load(preprocess_default_image, outputs=[image_input])

        source_mode.change(
            switch_asset_source,
            inputs=[source_mode],
            outputs=[
                source_tabs,
                asset_state,
                slat_upload,
                slat_path_text,
                mesh_viewer,
                pbr_viewer,
                color_output,
                base_color_output,
                metallic_output,
                roughness_output,
                shadow_output,
                camera_video_output,
                hdri_roll_video_output,
                hdri_render_video_output,
                status_md,
            ],
        )
        source_mode.change(
            lambda m: (
                gr.update(visible=m == "From Image"),
                gr.update(visible=m == "From Existing SLaT"),
            ),
            inputs=[source_mode],
            outputs=[col_img_examples, col_slat_examples],
        )

        for _trigger in (hdri_file.upload, hdri_file.change, tone_mapper_name.change):
            _trigger(
                preview_hdri,
                inputs=[hdri_file, tone_mapper_name],
                outputs=[hdri_preview, status_md],
            )

        # Same as TRELLIS.2 app.py: only on upload — avoids infinite preprocess loop.
        image_input.upload(
            preprocess_image_only,
            inputs=[image_input],
            outputs=[image_input],
        )

        mesh_button.click(
            generate_mesh,
            inputs=[image_input],
            outputs=[asset_state, mesh_viewer, status_md],
        )

        slat_button.click(
            prepare_slat,
            inputs=[source_mode, asset_state, image_input, seed, slat_upload, slat_path_text],
            outputs=[asset_state, status_md],
        )

        preview_button.click(
            render_preview,
            inputs=[asset_state, hdri_file, tone_mapper_name, hdri_rot,
                    yaw, pitch, fov, radius, resolution],
            outputs=[
                color_output,
                base_color_output,
                metallic_output,
                roughness_output,
                shadow_output,
                status_md,
            ],
        )

        camera_video_button.click(
            render_camera_video,
            inputs=[asset_state, hdri_file, tone_mapper_name, hdri_rot,
                    fps, num_views, fov, radius, full_video, shadow_video],
            outputs=[camera_video_output, status_md],
        )

        hdri_video_button.click(
            render_hdri_video,
            inputs=[asset_state, hdri_file, tone_mapper_name,
                    fps, num_frames, yaw, pitch, fov, radius, full_video, shadow_video],
            outputs=[hdri_roll_video_output, hdri_render_video_output, status_md],
        )

        export_glb_button.click(
            export_glb,
            inputs=[asset_state, hdri_file, tone_mapper_name, hdri_rot, simplify, texture_size],
            outputs=[pbr_viewer, status_md],
        )

        clear_button.click(
            clear_session_dir,
            outputs=[status_md],
        ).then(
            lambda: ({}, None, None, None, None, None, None, None, None, None, None),
            outputs=[
                asset_state,
                mesh_viewer,
                pbr_viewer,
                color_output,
                base_color_output,
                metallic_output,
                roughness_output,
                shadow_output,
                camera_video_output,
                hdri_roll_video_output,
                hdri_render_video_output,
            ],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_app()
    demo.queue(max_size=8)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.blue,
        ),
        css=CUSTOM_CSS,
    )
