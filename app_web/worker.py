import asyncio
from pathlib import Path
from typing import Callable, Any
from PIL import Image

from app_web.config import MOCK_MODE
from app_web.pipelines import PipelineManager

# Shared GPU lock to serialize PyTorch/CUDA operations across connections
gpu_lock = asyncio.Lock()

# Global pipeline manager instance
pipeline_manager = PipelineManager()


async def run_in_gpu_executor(func: Callable[..., Any], *args, **kwargs) -> Any:
    """Run a blocking GPU function in the default executor while holding the gpu_lock."""
    async with gpu_lock:
        loop = asyncio.get_running_loop()
        # Ensure models are loaded before executing if not in mock mode
        if not MOCK_MODE:
            await loop.run_in_executor(None, lambda: pipeline_manager.load_models())
            
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


async def generate_3d_asset_task(
    image_path: Path,
    seed: int,
    session_dir: Path,
    progress_callback: Callable[[float, str], None]
) -> tuple[Path, Path, Path]:
    """Complete 3D generation pipeline task:
    1. Generate geometry mesh (.glb)
    2. Generate SLaT representation (.npz)
    3. Render rotating 360 camera path video (.mp4)
    4. Render HDRI lighting rotation video (.mp4)
    
    Returns: (slat_path, video_path, hdri_video_path)
    """
    # Run the entire pipeline sequentially under the GPU lock
    async with gpu_lock:
        loop = asyncio.get_running_loop()
        
        if not MOCK_MODE:
            # Ensure models are loaded
            progress_callback(0.02, "Loading models into GPU...")
            await loop.run_in_executor(None, lambda: pipeline_manager.load_models())

        # Load input image
        image = Image.open(image_path)

        # Step 1: Generate Mesh
        def step1():
            return pipeline_manager.generate_mesh(
                image, 
                session_dir, 
                lambda p, m: progress_callback(0.02 + 0.38 * p, m)
            )
        mesh_path = await loop.run_in_executor(None, step1)

        # Step 2: Generate SLaT
        def step2():
            return pipeline_manager.generate_slat(
                image, 
                mesh_path, 
                seed, 
                session_dir, 
                lambda p, m: progress_callback(0.40 + 0.30 * p, m)
            )
        slat_path = await loop.run_in_executor(None, step2)

        # Step 3: Render rotating 360 camera path video
        # We use the default HDRI for the video rendering
        from app_web.config import DEFAULT_HDRI
        def step3():
            return pipeline_manager.render_camera_video(
                slat_path, 
                DEFAULT_HDRI, 
                session_dir, 
                lambda p, m: progress_callback(0.70 + 0.15 * p, m)
            )
        video_path = await loop.run_in_executor(None, step3)

        # Step 3b: Render HDRI lighting rotation video
        def step3b():
            return pipeline_manager.render_hdri_video(
                slat_path,
                DEFAULT_HDRI,
                session_dir,
                lambda p, m: progress_callback(0.85 + 0.15 * p, m)
            )
        hdri_video_path = await loop.run_in_executor(None, step3b)

        progress_callback(1.0, "Generation complete!")
        return slat_path, video_path, hdri_video_path


async def generate_style_transfer_task(
    geo_image_path: Path,
    style_image_path: Path,
    seed: int,
    session_dir: Path,
    progress_callback: Callable[[float, str], None]
) -> tuple[Path, Path, Path]:
    """Complete Style Transfer 3D generation pipeline task:
    1. Generate geometry mesh (.glb) from geo_image
    2. Generate SLaT representation (.npz) from style_image & geometry mesh
    3. Render rotating 360 camera path video (.mp4)
    4. Render HDRI lighting rotation video (.mp4)
    
    Returns: (slat_path, video_path, hdri_video_path)
    """
    async with gpu_lock:
        loop = asyncio.get_running_loop()
        
        if not MOCK_MODE:
            progress_callback(0.02, "Loading models into GPU...")
            await loop.run_in_executor(None, lambda: pipeline_manager.load_models())

        geo_image = Image.open(geo_image_path)
        style_image = Image.open(style_image_path)

        # Step 1: Generate Mesh from geo_image
        def step1():
            return pipeline_manager.generate_mesh(
                geo_image, 
                session_dir, 
                lambda p, m: progress_callback(0.02 + 0.38 * p, f"[Geometry] {m}")
            )
        mesh_path = await loop.run_in_executor(None, step1)

        # Step 2: Generate SLaT from style_image and mesh
        def step2():
            return pipeline_manager.generate_slat(
                style_image, 
                mesh_path, 
                seed, 
                session_dir, 
                lambda p, m: progress_callback(0.40 + 0.30 * p, f"[Style] {m}")
            )
        slat_path = await loop.run_in_executor(None, step2)

        # Step 3: Render rotating 360 camera path video
        from app_web.config import DEFAULT_HDRI
        def step3():
            return pipeline_manager.render_camera_video(
                slat_path, 
                DEFAULT_HDRI, 
                session_dir, 
                lambda p, m: progress_callback(0.70 + 0.15 * p, m)
            )
        video_path = await loop.run_in_executor(None, step3)

        # Step 3b: Render HDRI lighting rotation video
        def step3b():
            return pipeline_manager.render_hdri_video(
                slat_path,
                DEFAULT_HDRI,
                session_dir,
                lambda p, m: progress_callback(0.85 + 0.15 * p, m)
            )
        hdri_video_path = await loop.run_in_executor(None, step3b)

        progress_callback(1.0, "Style Transfer complete!")
        return slat_path, video_path, hdri_video_path


async def export_glb_task(
    slat_path: Path,
    hdri_path: Path,
    simplify: float,
    texture_size: int,
    session_dir: Path,
    progress_callback: Callable[[float, str], None]
) -> Path:
    """Bake PBR textures and export GLB task."""
    async with gpu_lock:
        loop = asyncio.get_running_loop()
        
        if not MOCK_MODE:
            await loop.run_in_executor(None, lambda: pipeline_manager.load_models())
            
        def run_export():
            return pipeline_manager.export_glb(
                slat_path,
                hdri_path,
                simplify,
                texture_size,
                session_dir,
                progress_callback
            )
            
        return await loop.run_in_executor(None, run_export)
