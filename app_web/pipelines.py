import os
import sys
import time
import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Callable
import numpy as np
import torch
from PIL import Image
import trimesh
import imageio

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

from app_web.config import MOCK_MODE, DEFAULT_SLAT, DEFAULT_HDRI, APP_DIR

# Add hy3dshape to path for Hunyuan3D
sys.path.insert(0, "./hy3dshape")

# Lazy imports to avoid importing heavy ML libraries on CPU-only machines
Hunyuan3DDiTFlowMatchingPipeline = None
NeARImageToRelightable3DPipeline = None

def import_ml_libraries():
    global Hunyuan3DDiTFlowMatchingPipeline, NeARImageToRelightable3DPipeline
    if Hunyuan3DDiTFlowMatchingPipeline is None:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # pyright: ignore[reportMissingImports]
    if NeARImageToRelightable3DPipeline is None:
        from trellis.pipelines import NeARImageToRelightable3DPipeline


def _ensure_rgba(img: Image.Image) -> Image.Image:
    """Normalize to RGBA so alpha is preserved for mesh (white matte) vs SLaT (black matte)."""
    if img.mode == "RGBA":
        return img
    if img.mode == "RGB":
        r, g, b = img.split()
        a = Image.new("L", img.size, 255)
        return Image.merge("RGBA", (r, g, b, a))
    return img.convert("RGBA")


def save_slat_npz(slat, save_path: Path):
    """Save SLaT representation as .npz."""
    np.savez(
        save_path,
        feats=slat.feats.detach().cpu().numpy(),
        coords=slat.coords.detach().cpu().numpy(),
    )


def compress_to_jpeg(rgb: np.ndarray, quality: int = 80) -> bytes:
    """Compress raw RGB numpy array to JPEG bytes."""
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def compress_to_png(rgb: np.ndarray, compression_level: int = 1) -> bytes:
    """Compress raw RGB numpy array to PNG bytes using OpenCV for maximum speed."""
    import cv2
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    success, encoded_img = cv2.imencode('.png', bgr, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
    if success:
        return encoded_img.tobytes()
    # Fallback to PIL
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=compression_level)
    return buf.getvalue()


def string_to_color(s: str) -> np.ndarray:
    """Generate a deterministic beautiful RGB color from a string."""
    h = 0
    for char in s:
        h = ord(char) + ((h << 5) - h)
    r = (h & 0xFF0000) >> 16
    g = (h & 0x00FF00) >> 8
    b = h & 0x0000FF
    color = np.array([r, g, b], dtype=np.float32) / 255.0
    color = 0.15 + 0.7 * color
    return color


def render_mock_sphere(
    yaw: float,
    pitch: float,
    radius: float,
    hdri_rot: float,
    res: int,
    tone_view: str = "AgX",
    slat_name: str = "",
    hdri_name: str = ""
) -> np.ndarray:
    """Render a beautiful, interactive 3D shaded sphere in CPU-only environments."""
    y, x = np.mgrid[-1:1:complex(0, res), -1:1:complex(0, res)]
    r2 = x**2 + y**2
    
    sphere_radius_sq = 0.55 / (radius ** 0.5)
    mask = r2 <= sphere_radius_sq
    
    # Pure black background
    rgb = np.zeros((res, res, 3), dtype=np.uint8)
    
    if np.any(mask):
        z = np.sqrt(np.maximum(0, sphere_radius_sq - r2))
        norm = np.stack([x, y, z], axis=-1)
        norm_len = np.linalg.norm(norm, axis=-1, keepdims=True)
        norm = np.where(norm_len > 0, norm / norm_len, 0)
        
        light_yaw = np.radians(hdri_rot)
        light_dir = np.array([np.cos(light_yaw), 0.5, np.sin(light_yaw)])
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        cy = np.cos(np.radians(yaw))
        sy = np.sin(np.radians(yaw))
        cp = np.cos(np.radians(pitch))
        sp = np.sin(np.radians(pitch))
        
        nx = norm[..., 0]
        ny = norm[..., 1]
        nz = norm[..., 2]
        
        rx = nx * cy - nz * sy
        rz = nx * sy + nz * cy
        ry = ny * cp - rz * sp
        rz = -ny * sp + rz * cp
        
        rotated_norm = np.stack([rx, ry, rz], axis=-1)
        diffuse = np.maximum(0, np.sum(rotated_norm * light_dir, axis=-1))
        
        view_dir = np.array([0, 0, 1])
        half_vec = (light_dir + view_dir)
        half_vec = half_vec / np.linalg.norm(half_vec)
        specular = np.maximum(0, np.sum(rotated_norm * half_vec, axis=-1)) ** 24
        
        base_color = string_to_color(slat_name) if slat_name else np.array([0.05, 0.55, 0.75])
        light_color = string_to_color(hdri_name + "_light") if hdri_name else np.array([1.0, 1.0, 1.0])
        
        ambient = 0.15
        shaded = (diffuse[..., None] * 0.85 + ambient) * base_color * light_color + specular[..., None] * 0.6
        
        if tone_view == "AgX":
            shaded = shaded / (shaded + 0.15) * 1.1
        elif tone_view in ("sRGB", "Standard"):
            shaded = np.clip(shaded, 0, 1) ** (1.0 / 2.2)
        
        shaded = np.clip(shaded * 255, 0, 255).astype(np.uint8)
        rgb[mask] = shaded[mask]
        
    return rgb


class PipelineManager:
    """Manager for loading and running Hunyuan3D and NeAR pipelines."""
    
    def __init__(self, pretrained_near: str = "luh0502/NeAR"):
        self.pretrained_near = pretrained_near
        self.geometry_pipeline = None
        self.pipeline = None
        self.sr_model = None
        
    def load_models(self):
        """Load the ML models into GPU memory if CUDA is available."""
        if MOCK_MODE:
            return
            
        import_ml_libraries()
        device = "cuda"
        
        if self.pipeline is None:
            print("[PipelineManager] Loading NeAR pipeline...", flush=True)
            self.pipeline = NeARImageToRelightable3DPipeline.from_pretrained(self.pretrained_near)
            self.pipeline.to(device)
            self.pipeline.setup_tone_mapper("AgX")
            
        if self.geometry_pipeline is None:
            print("[PipelineManager] Loading Hunyuan3D geometry pipeline...", flush=True)
            self.geometry_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
            self.geometry_pipeline.to(device)

        if self.sr_model is None:
            print("[PipelineManager] Loading DSCF Super-Resolution model...", flush=True)
            try:
                sys.path.insert(0, str(APP_DIR / "DSCF-SR"))
                from models.team23_DSCF import DSCF
                self.sr_model = DSCF(num_in_ch=3, num_out_ch=3, feature_channels=26, upscale=4)
                sr_weight_path = APP_DIR / "DSCF-SR/model_zoo/team23_DSCF.pth"
                if sr_weight_path.exists():
                    state_dict = torch.load(sr_weight_path, map_location=device)
                    self.sr_model.load_state_dict(state_dict, strict=False)
                    self.sr_model.eval()
                    self.sr_model.to(device)
                    print("[PipelineManager] DSCF Super-Resolution model loaded successfully!", flush=True)
                else:
                    print(f"[PipelineManager] Warning: Super-Resolution weights not found at {sr_weight_path}", flush=True)
                    self.sr_model = None
            except Exception as e:
                print(f"[PipelineManager] Error loading DSCF-SR model: {e}", flush=True)
                self.sr_model = None

    @torch.no_grad()
    def super_resolve_numpy(self, rgb_np: np.ndarray) -> np.ndarray:
        """Upscale a uint8 (H, W, 3) numpy array 4x using DSCF-SR on GPU."""
        if self.sr_model is None:
            return rgb_np
            
        try:
            # 1. Convert to float tensor [0, 1] on GPU
            tensor_in = torch.from_numpy(rgb_np.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to("cuda")
            
            # 2. Forward pass through DSCF-SR
            tensor_out = self.sr_model(tensor_in)
            
            # 3. Postprocess back to uint8 numpy array
            tensor_out = tensor_out.clamp(0.0, 1.0).squeeze(0).cpu()
            rgb_out = (tensor_out.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
            return rgb_out
        except Exception as e:
            print(f"[PipelineManager] Super-resolution failed: {e}")
            return rgb_np

    def generate_mesh(
        self, 
        image_input: Image.Image, 
        session_dir: Path, 
        progress_callback: Callable[[float, str], None]
    ) -> Path:
        """Step 1: Generate Hunyuan3D geometry from input image."""
        mesh_path = session_dir / "initial_3d_shape.glb"
        
        if MOCK_MODE:
            progress_callback(0.2, "Removing background (Mock)...")
            time.sleep(1.0)
            progress_callback(0.6, "Generating geometry (Mock)...")
            time.sleep(1.5)
            
            # Create a nice mock GLB sphere
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
            mesh.export(mesh_path)
            progress_callback(1.0, "Mesh ready!")
            return mesh_path

        self.load_models()
        
        progress_callback(0.2, "Removing background...")
        rgba = _ensure_rgba(image_input)
        if rgba.size != (518, 518):
            rgba = self.pipeline.preprocess_image_rgba(rgba)
            
        mesh_rgb = self.pipeline.flatten_rgba_on_matte(rgba, (1.0, 1.0, 1.0))
        rgba.save(session_dir / "input_preprocessed_rgba.png")
        mesh_rgb.save(session_dir / "input_processed.png")
        
        progress_callback(0.6, "Generating geometry...")
        with torch.inference_mode():
            mesh = self.geometry_pipeline(image=mesh_rgb)[0]
            
        mesh.export(mesh_path)
        progress_callback(1.0, "Mesh ready!")
        return mesh_path

    def generate_slat(
        self, 
        image_input: Image.Image, 
        mesh_path: Path, 
        seed: int, 
        session_dir: Path, 
        progress_callback: Callable[[float, str], None]
    ) -> Path:
        """Step 2: Generate SLaT representation from mesh and image."""
        slat_path = session_dir / "generated_slat.npz"
        
        if MOCK_MODE:
            progress_callback(0.3, "Computing SLaT coordinates (Mock)...")
            time.sleep(1.0)
            progress_callback(0.7, "Generating SLaT (Mock)...")
            time.sleep(1.5)
            
            # Save dummy SLaT npz
            if DEFAULT_SLAT.exists():
                import shutil
                shutil.copy(DEFAULT_SLAT, slat_path)
            else:
                np.savez(slat_path, feats=np.zeros((100, 16)), coords=np.zeros((100, 4)))
            progress_callback(1.0, "SLaT generated!")
            return slat_path

        self.load_models()
        
        progress_callback(0.2, "Loading mesh...")
        mesh = trimesh.load(mesh_path, force="mesh")
        rgba = _ensure_rgba(image_input)
        if rgba.size != (518, 518):
            rgba = self.pipeline.preprocess_image_rgba(rgba)
        slat_rgb = self.pipeline.flatten_rgba_on_matte(rgba, (0.0, 0.0, 0.0))
        
        progress_callback(0.5, "Computing SLaT coordinates...")
        coords = self.pipeline.shape_to_coords(mesh)
        
        progress_callback(0.8, "Generating SLaT...")
        with torch.inference_mode():
            slat = self.pipeline.run_with_coords([slat_rgb], coords, seed=int(seed), preprocess_image=False)
            
        save_slat_npz(slat, slat_path)
        progress_callback(1.0, "SLaT generated!")
        return slat_path

    def render_camera_video(
        self, 
        slat_path: Path, 
        hdri_path: Path, 
        session_dir: Path, 
        progress_callback: Callable[[float, str], None]
    ) -> Path:
        """Step 3: Render rotating 360 camera path video."""
        video_path = session_dir / f"camera_path_{time.time_ns()}.mp4"
        
        if MOCK_MODE:
            progress_callback(0.5, "Rendering camera path (Mock)...")
            time.sleep(1.5)
            
            # Create a simple 10-frame rotating mock video (e.g., color-shifting frames)
            frames = []
            for i in range(20):
                frame = np.zeros((256, 256, 3), dtype=np.uint8)
                center_x = int(128 + 40 * np.cos(i * np.pi / 10))
                center_y = int(128 + 40 * np.sin(i * np.pi / 10))
                for y in range(256):
                    for x in range(256):
                        if (x - center_x)**2 + (y - center_y)**2 < 30**2:
                            frame[y, x] = [50, 180, 220]
                        else:
                            frame[y, x] = [15, 15, 20]
                frames.append(frame)
                
            imageio.mimsave(video_path, frames, fps=15)
            progress_callback(1.0, "Video rendered!")
            return video_path

        self.load_models()
        
        progress_callback(0.2, "Loading SLaT and HDRI...")
        slat = self.pipeline.load_slat(str(slat_path))
        hdri_np = self.pipeline.load_hdri(str(hdri_path))
        
        progress_callback(0.6, "Rendering camera path video...")
        with torch.inference_mode():
            frames = self.pipeline.render_camera_path_video(
                slat, hdri_np,
                num_views=40, fov=40.0, radius=2.0,
                hdri_rot_deg=0.0, full_video=True, shadow_video=True,
                bg_color=(0, 0, 0), verbose=False,
            )
            
        imageio.mimsave(video_path, frames, fps=15)
        progress_callback(1.0, "Video rendered!")
        return video_path

    def render_hdri_video(
        self, 
        slat_path: Path, 
        hdri_path: Path, 
        session_dir: Path, 
        progress_callback: Callable[[float, str], None]
    ) -> Path:
        """Step 3b: Render rotating 360 HDRI illumination rotation video."""
        video_path = session_dir / f"hdri_rotation_{time.time_ns()}.mp4"
        
        if MOCK_MODE:
            progress_callback(0.5, "Rendering HDRI rotation path (Mock)...")
            time.sleep(1.5)
            
            frames = []
            for i in range(20):
                frame = np.zeros((256, 256, 3), dtype=np.uint8)
                center_x = 128
                center_y = 128
                for y in range(256):
                    for x in range(256):
                        if (x - center_x)**2 + (y - center_y)**2 < 30**2:
                            # Dynamic color matching rotating light in mock mode
                            shift = int(20 * np.sin(i * np.pi / 10))
                            frame[y, x] = [50 + shift, 180 + shift, 220]
                        else:
                            frame[y, x] = [5, 5, 5]
                frames.append(frame)
                
            imageio.mimsave(video_path, frames, fps=15)
            progress_callback(1.0, "HDRI video rendered!")
            return video_path

        self.load_models()
        
        progress_callback(0.2, "Loading SLaT and HDRI...")
        slat = self.pipeline.load_slat(str(slat_path))
        hdri_np = self.pipeline.load_hdri(str(hdri_path))
        
        progress_callback(0.6, "Rendering HDRI rotation video...")
        with torch.inference_mode():
            # returns: hdri_roll_frames, render_frames
            _, render_frames = self.pipeline.render_hdri_rotation_video(
                slat, hdri_np,
                num_frames=40, yaw_deg=0.0, pitch_deg=0.0,
                fov=40.0, radius=2.0, full_video=True, shadow_video=True,
                bg_color=(0, 0, 0), verbose=False,
            )
            
        imageio.mimsave(video_path, render_frames, fps=15)
        progress_callback(1.0, "HDRI video rendered!")
        return video_path

    def export_glb(
        self, 
        slat_path: Path, 
        hdri_path: Path, 
        simplify: float, 
        texture_size: int, 
        session_dir: Path, 
        progress_callback: Callable[[float, str], None]
    ) -> Path:
        """Bake PBR textures and export a fully textured .glb file."""
        glb_path = session_dir / f"near_pbr_{time.time_ns()}.glb"
        
        if MOCK_MODE:
            progress_callback(0.5, "Baking PBR textures (Mock)...")
            time.sleep(2.0)
            
            # Export a simple sphere as mock GLB
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
            mesh.export(glb_path)
            progress_callback(1.0, "PBR GLB exported!")
            return glb_path

        self.load_models()
        
        progress_callback(0.2, "Loading SLaT and HDRI...")
        slat = self.pipeline.load_slat(str(slat_path))
        hdri_np = self.pipeline.load_hdri(str(hdri_path))
        
        progress_callback(0.6, "Baking PBR textures and exporting GLB...")
        glb = self.pipeline.export_glb_from_slat(
            slat, hdri_np,
            hdri_rot_deg=0.0, base_mesh=None,
            simplify=simplify, texture_size=int(texture_size), fill_holes=True,
        )
            
        glb.export(glb_path)
        progress_callback(1.0, "PBR GLB exported!")
        return glb_path
