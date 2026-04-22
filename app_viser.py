#!/usr/bin/env python3
"""NeAR neural relighting viewer using viser.

Server-side SLatGaussianRenderer + gsplat; the browser shows only the relit RGB image as a
full-viewport background (``SceneApi.set_background_image``). Orbit the camera to change
yaw / pitch / distance / vertical FOV.

Run from the NeAR repository root (same as ``app.py``)::

    cd /path/to/NeAR
    python app_viser.py --slat assets/example_slats/<id>.npz --hdri assets/hdris/studio_small_03_1k.exr

Requires CUDA. Install viser with ``pip install 'viser>=1.0.0'`` (also added by ``setup.sh --demo``).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "./hy3dshape")
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")  # H series; override if needed

try:
    import viser
except ImportError as e:
    raise SystemExit(
        "viser is required. Install with: pip install 'viser>=1.0.0'\n" f"({e})"
    ) from e

from trellis.pipelines import NeARImageToRelightable3DPipeline


APP_DIR = Path(__file__).resolve().parent
DEFAULT_HDRI = APP_DIR / "assets/hdris/studio_small_03_1k.exr"
DEFAULT_SLAT = APP_DIR / (
    "assets/example_slats/"
    "2a0d671ce308adb93323eae7141953fc1a5ba68f38cc69f476d5e904c634864d.npz"
)

_gpu_render_lock = threading.Lock()


def _letterbox_square_to_viewport(
    rgb: np.ndarray,
    viewport_w: int,
    viewport_h: int,
    bg_rgb: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Place a square render (S,S,3) centered on a (H,W,3) canvas matching the browser viewport.

    Without this, viser stretches the square buffer to the full viewport and the subject looks
    horizontally stretched or vertically squashed on wide screens.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return rgb
    if viewport_w <= 0 or viewport_h <= 0:
        return rgb
    s = int(rgb.shape[0])
    if s <= 0 or rgb.shape[1] != s:
        return rgb

    side = int(min(viewport_w, viewport_h))
    if side < 1:
        return rgb

    if side != s:
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x = F.interpolate(x, size=(side, side), mode="bilinear", align_corners=False)
        tile = (x.squeeze(0).permute(1, 2, 0) * 255.0).clamp(0, 255).to(torch.uint8).numpy()
    else:
        tile = rgb

    canvas = np.empty((viewport_h, viewport_w, 3), dtype=np.uint8)
    canvas[:, :] = np.array(bg_rgb, dtype=np.uint8)
    x0 = (viewport_w - side) // 2
    y0 = (viewport_h - side) // 2
    canvas[y0 : y0 + side, x0 : x0 + side] = tile
    return canvas


@dataclass
class _ClientRenderState:
    idle_timer: Optional[threading.Timer] = None
    last_drag_render_t: float = 0.0


def _viser_camera_to_near_orbit(cam: Any) -> Tuple[float, float, float, float]:
    """Map viser orbit camera to NeAR yaw, pitch (degrees), radius, vertical FOV (degrees)."""
    pos = np.asarray(cam.position, dtype=np.float64).reshape(3)
    r = float(np.linalg.norm(pos))
    if r < 1e-4:
        return 0.0, 0.0, 2.0, 40.0
    u = pos / r
    uz = float(np.clip(u[2], -1.0, 1.0))
    pitch_deg = float(np.rad2deg(np.arcsin(uz)))
    cp = float(np.cos(np.deg2rad(pitch_deg)))
    if abs(cp) > 1e-5:
        yaw_deg = float(np.rad2deg(np.arctan2(float(u[0]), float(u[1]))))
    else:
        yaw_deg = 0.0
    fov_deg = float(np.clip(np.rad2deg(float(cam.fov)), 10.0, 89.0))
    r = float(np.clip(r, 0.25, 8.0))
    return yaw_deg, pitch_deg, r, fov_deg


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeAR neural relight viewer (viser)")
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
        "--drag-res",
        type=int,
        default=256,
        help="Internal render resolution while the camera is moving",
    )
    p.add_argument(
        "--idle-res",
        type=int,
        default=512,
        help="Internal render resolution after the camera stops (debounced)",
    )
    p.add_argument(
        "--throttle-s",
        type=float,
        default=0.12,
        help="Minimum seconds between low-res renders during motion",
    )
    p.add_argument(
        "--idle-delay-s",
        type=float,
        default=0.45,
        help="Seconds after the last camera move before an idle-res render",
    )
    p.add_argument(
        "--share",
        action="store_true",
        help="After startup, request a public share URL (viser tunnel; experimental, may block briefly)",
    )
    p.add_argument(
        "--no-letterbox",
        action="store_true",
        help="Stretch square render to full viewport (old behavior; distorts aspect ratio)",
    )
    p.add_argument(
        "--clip-near",
        type=float,
        default=0.05,
        help="Neural renderer near clip (view-space); must stay below orbit radius when zoomed in",
    )
    p.add_argument(
        "--clip-far",
        type=float,
        default=32.0,
        help="Neural renderer far clip (view-space); increase if the object vanishes when zooming out",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not (0.0 < float(args.clip_near) < float(args.clip_far)):
        raise SystemExit(
            f"Need 0 < --clip-near < --clip-far; got near={args.clip_near!r} far={args.clip_far!r}"
        )
    if not args.slat or not Path(args.slat).is_file():
        raise SystemExit(f"Missing or invalid --slat: {args.slat!r}")
    if not args.hdri or not Path(args.hdri).is_file():
        raise SystemExit(f"Missing or invalid --hdri: {args.hdri!r}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for NeAR neural rendering.")

    print("[app_viser] Loading NeAR pipeline...", flush=True)
    pipeline = NeARImageToRelightable3DPipeline.from_pretrained(args.pretrained)
    pipeline.to("cuda")

    available_tm = getattr(pipeline.tone_mapper, "available_views", ["AgX"])
    tone_default = available_tm[0] if available_tm else "AgX"
    pipeline.setup_tone_mapper(tone_default)

    print("[app_viser] Loading SLaT and HDRI...", flush=True)
    slat = pipeline.load_slat(args.slat)
    hdri_np = pipeline.load_hdri(args.hdri)

    with torch.inference_mode():
        hs, rfs = pipeline.decoder_pbr_feats(slat)

    shared: Dict[str, Any] = {
        "hdri_rot_deg": 0.0,
        "tone_view": tone_default,
        "hdri_cond": None,
    }
    client_states: Dict[int, _ClientRenderState] = {}

    def _refresh_hdri_cond() -> None:
        with torch.inference_mode():
            shared["hdri_cond"] = pipeline.encode_hdri(hdri_np, shared["hdri_rot_deg"])

    _refresh_hdri_cond()

    def _cancel_idle(client_id: int) -> None:
        st = client_states.get(client_id)
        if st is None or st.idle_timer is None:
            return
        st.idle_timer.cancel()
        st.idle_timer = None

    def _render_client(client: viser.ClientHandle, internal_res: int) -> None:
        yaw_deg, pitch_deg, radius, fov_deg = _viser_camera_to_near_orbit(client.camera)
        with _gpu_render_lock:
            pipeline.setup_tone_mapper(shared["tone_view"])
            hc = shared["hdri_cond"]
            if hc is None:
                _refresh_hdri_cond()
                hc = shared["hdri_cond"]
            rgb = pipeline.render_relight_color_numpy(
                hs,
                rfs,
                hc,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                fov_deg=fov_deg,
                radius=radius,
                internal_resolution=internal_res,
                bg_color=(1.0, 1.0, 1.0),
                clip_near=float(args.clip_near),
                clip_far=float(args.clip_far),
            )
        if not args.no_letterbox:
            try:
                vw = int(client.camera.image_width)
                vh = int(client.camera.image_height)
            except Exception:
                vw, vh = 0, 0
            if vw > 0 and vh > 0:
                rgb = _letterbox_square_to_viewport(rgb, vw, vh, bg_rgb=(255, 255, 255))
        with client.atomic():
            client.scene.set_background_image(rgb, format="jpeg")
            client.flush()

    def _schedule_idle_render(client: viser.ClientHandle, idle_res: int, delay_s: float) -> None:
        cid = client.client_id
        st = client_states.setdefault(cid, _ClientRenderState())
        _cancel_idle(cid)

        def _fire() -> None:
            st.idle_timer = None
            try:
                _render_client(client, idle_res)
            except Exception as ex:
                client.add_notification("Render failed", str(ex), color="red")

        t = threading.Timer(max(0.05, delay_s), _fire)
        st.idle_timer = t
        t.daemon = True
        t.start()

    server = viser.ViserServer(host=args.host, port=args.port, label="NeAR relight")
    server.scene.set_global_visibility(False)

    def _request_share_url(notify_client: Optional[Any] = None) -> None:
        """Experimental public URL via viser tunnel; may block on first call."""
        fn = getattr(server, "request_share_url", None)
        if fn is None:
            msg = "request_share_url() not available in this viser version"
            print(f"[app_viser] {msg}", flush=True)
            if notify_client is not None:
                notify_client.add_notification("Share", msg, color="red")
            return
        try:
            url = fn(verbose=True)
        except Exception as ex:
            print(f"[app_viser] Share failed: {ex}", flush=True)
            if notify_client is not None:
                notify_client.add_notification("Share failed", str(ex), color="red")
            return
        if url:
            print(f"[app_viser] Share URL: {url}", flush=True)
            if notify_client is not None:
                notify_client.add_notification("Share URL", str(url), color="blue")
        else:
            print("[app_viser] Share URL unavailable (None)", flush=True)
            if notify_client is not None:
                notify_client.add_notification("Share", "URL unavailable", color="red")

    server.initial_camera.position = (0.0, 2.0, 0.5)
    server.initial_camera.look_at = (0.0, 0.0, 0.0)
    server.initial_camera.fov = float(np.deg2rad(40.0))

    gui = server.gui
    hdri_rot = gui.add_slider(
        "HDRI rotation (deg)",
        0.0,
        360.0,
        1.0,
        0.0,
    )
    tone_dd = gui.add_dropdown(
        "Tone mapper",
        options=list(available_tm),
        initial_value=tone_default,
    )
    drag_res_slider = gui.add_slider(
        "Drag render res",
        64,
        512,
        32,
        float(args.drag_res),
    )
    idle_res_slider = gui.add_slider(
        "Idle render res",
        128,
        1024,
        64,
        float(args.idle_res),
    )
    refresh_btn = gui.add_button("Render now (idle res)")
    share_btn = gui.add_button("Get share URL (tunnel)")

    @hdri_rot.on_update
    def _on_hdri_rot(_: Any) -> None:
        shared["hdri_rot_deg"] = float(hdri_rot.value)
        _refresh_hdri_cond()
        for c in server.get_clients().values():
            _schedule_idle_render(c, int(idle_res_slider.value), 0.05)

    @tone_dd.on_update
    def _on_tone(_: Any) -> None:
        shared["tone_view"] = str(tone_dd.value)
        pipeline.setup_tone_mapper(shared["tone_view"])
        for c in server.get_clients().values():
            _schedule_idle_render(c, int(idle_res_slider.value), 0.05)

    @refresh_btn.on_click
    def _on_refresh(event: viser.GuiEvent) -> None:
        client = event.client
        if client is None:
            return
        try:
            _render_client(client, int(idle_res_slider.value))
        except Exception as ex:
            client.add_notification("Render failed", str(ex), color="red")

    @share_btn.on_click
    def _on_share(event: viser.GuiEvent) -> None:
        client = event.client
        if client is None:
            return
        client.add_notification("Share", "Requesting tunnel URL…", color="blue")

        def _worker() -> None:
            _request_share_url(client)

        threading.Thread(target=_worker, daemon=True).start()

    @server.on_client_connect
    def _on_connect(client: viser.ClientHandle) -> None:
        client.scene.set_global_visibility(False)
        client.camera.look_at = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        client_states[client.client_id] = _ClientRenderState()
        try:
            _render_client(client, int(idle_res_slider.value))
        except Exception as ex:
            client.add_notification("Initial render failed", str(ex), color="red")

        throttle = max(0.05, float(args.throttle_s))
        idle_delay = max(0.05, float(args.idle_delay_s))

        @client.camera.on_update
        def _on_cam(_cam: viser.CameraHandle) -> None:
            st = client_states.setdefault(client.client_id, _ClientRenderState())
            now = time.time()
            drag_res = int(drag_res_slider.value)
            if now - st.last_drag_render_t >= throttle:
                try:
                    _render_client(client, drag_res)
                except Exception:
                    pass
                st.last_drag_render_t = now
            _schedule_idle_render(client, int(idle_res_slider.value), idle_delay)

    @server.on_client_disconnect
    def _on_disconnect(client: viser.ClientHandle) -> None:
        _cancel_idle(client.client_id)
        client_states.pop(client.client_id, None)

    host_disp = server.get_host()
    port_disp = server.get_port()
    print(
        f"[app_viser] Listening on http://{host_disp}:{port_disp} "
        f"(also try http://127.0.0.1:{port_disp})",
        flush=True,
    )
    if args.share:

        def _share_at_startup() -> None:
            time.sleep(2.0)
            _request_share_url(None)

        threading.Thread(target=_share_at_startup, daemon=True).start()
    server.sleep_forever()


if __name__ == "__main__":
    main()
