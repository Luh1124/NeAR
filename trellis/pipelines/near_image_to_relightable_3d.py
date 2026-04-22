from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union, Callable

import cv2
import numpy as np
import open3d as o3d
import pyexr
import rembg
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from easydict import EasyDict as edict
from PIL import Image
from simple_ocio import ToneMapper
from torchvision import transforms

from . import samplers, rembg
from .base import Pipeline
from ..datasets.hdri_processer import HDRI_Preprocessor
from ..modules import sparse as sp
from ..utils import render_utils_rl


ImageInput = Union[torch.Tensor, List[Image.Image]]
SparseTensor = sp.SparseTensor


class NeARImageToRelightable3DPipeline(Pipeline):
    """Pipeline for relightable 3D asset generation and rendering."""

    def __init__(
        self,
        models: Optional[Dict[str, nn.Module]] = None,
        sparse_structure_sampler: Optional[samplers.Sampler] = None,
        slat_sampler: Optional[samplers.Sampler] = None,
        slat_normalization: Optional[dict] = None,
        image_cond_model: Optional[str] = None,
        rembg_model: Optional[Callable] = None,
    ):
        if models is None:
            return

        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.slat_sampler_params: dict = {}
        self.slat_normalization = slat_normalization
        self.rembg_model = rembg_model
        self._init_image_cond_model(image_cond_model)
        self.hdri_processor = HDRI_Preprocessor(envmap_h=512, envmap_w=1024)
        self.renderer = None
        self.tone_mapper = None

    def setup_renderer(
        self,
        resolution: int = 512,
        near: float = 1,
        far: float = 3,
        bg_color: Tuple[float, float, float] = (0, 0, 0),
        ssaa: int = 1,
    ) -> None:
        """Initialize the renderer used for image and video rendering."""
        self.renderer = render_utils_rl.get_renderer(
            resolution=resolution,
            near=near,
            far=far,
            bg_color=bg_color,
            ssaa=ssaa,
        )

    def setup_tone_mapper(self, view: str = "AgX") -> None:
        """Initialize the tone mapper used for HDR-to-LDR conversion."""
        self.tone_mapper = ToneMapper()
        self.tone_mapper.view = view

    @staticmethod
    def from_pretrained(path: str) -> "NeARImageToRelightable3DPipeline":
        """Load a pretrained NeAR pipeline from a local path or HF repository."""
        pipeline = super(
            NeARImageToRelightable3DPipeline,
            NeARImageToRelightable3DPipeline,
        ).from_pretrained(path)
        new_pipeline = NeARImageToRelightable3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__

        args = pipeline._pretrained_args
        new_pipeline.slat_sampler = getattr(samplers, args["slat_sampler"]["name"])(
            **args["slat_sampler"]["args"]
        )
        new_pipeline.slat_sampler_params = args["slat_sampler"]["params"]
        new_pipeline.slat_normalization = args["slat_normalization"]
        new_pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        new_pipeline._init_image_cond_model(args["image_cond_model"])
        new_pipeline.hdri_processor = HDRI_Preprocessor(envmap_h=512, envmap_w=1024)
        new_pipeline.setup_renderer()
        new_pipeline.setup_tone_mapper("AgX")
        return new_pipeline

    def _init_image_cond_model(self, name: str) -> None:
        """Initialize the image-conditioning backbone and normalization."""
        dinov2_model = torch.hub.load("facebookresearch/dinov2", name, pretrained=True).eval()
        self.models["image_cond_model"] = dinov2_model
        self.image_cond_model_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def preprocess_image_rgba(self, input: Image.Image) -> Image.Image:
        """Remove background when needed, crop, and resize to 518²; returns RGBA."""
        has_alpha = False
        if input.mode == "RGBA":
            alpha = np.array(input)[:, :, 3]
            has_alpha = not np.all(alpha == 255)

        if has_alpha:
            output = input
        else:
            input = input.convert("RGB")
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize(
                    (int(input.width * scale), int(input.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            output = self.rembg_model(input)

        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = (
            np.min(bbox[:, 1]),
            np.min(bbox[:, 0]),
            np.max(bbox[:, 1]),
            np.max(bbox[:, 0]),
        )
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        crop_bbox = (
            center[0] - size // 2,
            center[1] - size // 2,
            center[0] + size // 2,
            center[1] + size // 2,
        )

        output = output.crop(crop_bbox)  # type: ignore[arg-type]
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        if output.mode != "RGBA":
            output = output.convert("RGBA")
        return output

    @staticmethod
    def flatten_rgba_on_matte(
        image: Image.Image, matte_rgb: Tuple[float, float, float]
    ) -> Image.Image:
        """Composite RGBA onto a solid background; opaque RGB is returned unchanged."""
        if image.mode != "RGBA":
            return image.convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        rgb, a = arr[:, :, :3], arr[:, :, 3:4]
        m = np.array(matte_rgb, dtype=np.float32).reshape(1, 1, 3)
        out = rgb * a + (1.0 - a) * m
        return Image.fromarray((np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8))

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """Historical behavior: same crop as preprocess_image_rgba, matted on black."""
        rgba = self.preprocess_image_rgba(input)
        return self.flatten_rgba_on_matte(rgba, (0.0, 0.0, 0.0))

    def process_hdri(self, hdri: np.ndarray, hdri_rot: float = 0) -> torch.Tensor:
        """Convert an HDRI image into the conditioning tensor expected by the model."""
        hdri = cv2.resize(hdri, (1024, 512), interpolation=cv2.INTER_NEAREST)
        hdri = torch.from_numpy(hdri)
        hdri_rot = [0, 0, hdri_rot * np.pi / 180]
        envir_map_ldr, envir_map_hdr, _, _, view_dirs_world = self.hdri_processor.preprcess_envir_map(
            hdri,
            hdri_rot,
        )
        return torch.cat([envir_map_ldr, envir_map_hdr, view_dirs_world], dim=0).float()

    @torch.no_grad()
    def encode_image(self, image: ImageInput) -> torch.Tensor:
        """Encode input image(s) with the image-conditioning backbone."""
        if isinstance(image, torch.Tensor):
            if image.ndim != 4:
                raise AssertionError("Image tensor should be batched (B, C, H, W)")
        elif isinstance(image, list):
            if not all(isinstance(i, Image.Image) for i in image):
                raise AssertionError("Image list should contain PIL images")
            image = [img.resize((518, 518), Image.LANCZOS) for img in image]
            image = [np.array(img.convert("RGB")).astype(np.float32) / 255 for img in image]
            image = [torch.from_numpy(img).permute(2, 0, 1).float() for img in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models["image_cond_model"](image, is_training=True)["x_prenorm"]
        return F.layer_norm(features, features.shape[-1:])

    def get_cond(self, image: ImageInput) -> dict:
        """Create positive and negative image-conditioning features."""
        cond = self.encode_image(image)
        return {
            "cond": cond,
            "neg_cond": torch.zeros_like(cond),
        }

    def decoder_pbr_feats(self, slat: SparseTensor) -> Tuple[SparseTensor, torch.Tensor]:
        """Decode structured latents into renderer features."""
        return self.models["decoder"](slat)

    def render_gs(
        self,
        slat: SparseTensor,
        reg_feats: torch.Tensor,
        hdri_cond: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> Any:
        """Render Gaussian-splat-like representations from decoded features."""
        return self.models["renderer"](slat, reg_feats, hdri_cond, extrinsics)

    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """Sample a structured latent conditioned on image features and sparse coords."""
        flow_model = self.models["slat_flow_model"]
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
        ).samples

        std = torch.tensor(self.slat_normalization["std"])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization["mean"])[None].to(slat.device)
        return slat * std + mean

    @torch.no_grad()
    def shape_to_coords(self, mesh: trimesh.Trimesh) -> torch.Tensor:
        """Voxelize a mesh into sparse coordinates used by the SLaT sampler."""
        transform_y_to_z_up = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        mesh = mesh.apply_transform(transform_y_to_z_up)

        o3d_mesh = o3d.geometry.TriangleMesh()
        vertices = np.clip(np.asarray(mesh.vertices * 0.5), -0.5 + 1e-6, 0.5 - 1e-6)
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            o3d_mesh,
            voxel_size=1 / 64,
            min_bound=(-0.5, -0.5, -0.5),
            max_bound=(0.5, 0.5, 0.5),
        )
        vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        indices = torch.from_numpy(vertices + 0.5).to(torch.int32)
        batch_col = torch.zeros(indices.shape[0], 1, dtype=torch.int32)
        return torch.cat([batch_col, indices], dim=1).to(self.device)

    @torch.no_grad()
    def run_with_shape(
        self,
        image: Image.Image,
        mesh: trimesh.Trimesh,
        seed: int = 42,
        slat_sampler_params: dict = {},
        preprocess_image: bool = True,
    ) -> Any:
        """Infer a SLaT from a single image and an external mesh prior."""
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        coords = self.shape_to_coords(mesh)
        return self.sample_slat(cond, coords, slat_sampler_params)

    @torch.no_grad()
    def run_with_coords(
        self,
        image: Image.Image,
        coords: torch.Tensor,
        seed: int = 42,
        preprocess_image: bool = True,
        slat_sampler_params: dict = {},
    ) -> Any:
        """Infer a SLaT from image conditions and externally provided sparse coords."""
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond(image)
        torch.manual_seed(seed)
        return self.sample_slat(cond, coords.to(self.device), slat_sampler_params)

    @torch.no_grad()
    def encode_hdri(self, hdri_cond: np.ndarray, hdri_rot: float = 0) -> torch.Tensor:
        """Encode an HDRI map into the lighting token representation."""
        hdri_cond = self.process_hdri(hdri_cond, hdri_rot)
        return self.models["hdri_encoder"](hdri_cond[None].to(self.device))

    @torch.no_grad()
    def get_reps(
        self,
        hs: SparseTensor,
        rfs: torch.Tensor,
        hdri_cond: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> Any:
        """Query the renderer backbone for scene representations."""
        return self.models["renderer"](hs, rfs, hdri_cond, extrinsics)

    def load_slat(self, slat_path: str) -> SparseTensor:
        """Load a SLaT from `.npz` with keys `feats` and `coords`."""
        loaded = np.load(slat_path, allow_pickle=True)
        feats = torch.from_numpy(loaded["feats"]).to(torch.float32)
        coords = torch.from_numpy(loaded["coords"]).to(torch.int32)
        if coords.shape[-1] == 3:
            coords = torch.cat(
                [
                    torch.zeros((coords.shape[0], 1), device=coords.device, dtype=coords.dtype),
                    coords,
                ],
                dim=-1,
            )
        return sp.SparseTensor(feats, coords).to(self.device)

    def load_hdri(self, hdri_path: str) -> np.ndarray:
        """Load an EXR HDRI file as a float array with shape `(H, W, 3)`."""
        return pyexr.read(hdri_path)[..., :3]

    def generate_camera(
        self,
        yaw_deg: float,
        pitch_deg: float,
        radius: float,
        fov: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a single camera extrinsic / intrinsic pair from view parameters."""
        yaw_rad = np.deg2rad(float(yaw_deg))
        pitch_rad = np.deg2rad(float(pitch_deg))
        extr, intr = render_utils_rl.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            [yaw_rad],
            [pitch_rad],
            radius,
            fov,
        )
        return extr[0].to(self.device), intr[0].to(self.device)

    def generate_spiral_cameras(
        self,
        num_views: int,
        radius: float = 2.0,
        fov: float = 40.0,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Generate a spiral camera path for video rendering."""
        extrinsics, intrinsics = render_utils_rl.generate_cameras_spiral(num_views, r=radius, fov=fov)
        extrinsics = [e.to(self.device) for e in extrinsics]
        intrinsics = [i.to(self.device) for i in intrinsics]
        return extrinsics, intrinsics

    @staticmethod
    def _to_pil_with_alpha(x: np.ndarray, alpha_ch: np.ndarray) -> Image.Image:
        """Convert a predicted map and alpha channel into an RGBA PIL image."""
        if x.shape[-1] == 1:
            x = x.repeat(3, axis=-1)
        x = np.concatenate([x, alpha_ch], axis=-1)
        return Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))

    @torch.no_grad()
    def render_view(
        self,
        slat: SparseTensor,
        hdri_np: np.ndarray,
        yaw_deg: float,
        pitch_deg: float,
        fov: float = 40.0,
        radius: float = 2.0,
        hdri_rot_deg: float = 0.0,
        resolution: int = 1024,
    ) -> Dict[str, Image.Image]:
        """Render a single relit view and return color/material/shadow maps."""
        hdri_cond = self.encode_hdri(hdri_np, hdri_rot_deg)
        hs, rfs = self.decoder_pbr_feats(slat)
        extr, intr = self.generate_camera(yaw_deg, pitch_deg, radius, fov)
        reps = self.get_reps(hs, rfs, hdri_cond, extr[None, ...])

        res = None
        for rep in reps:
            res = self.renderer.render(rep, extr, intr, opt=edict(neural_basis=self.models["neural_basis"]))
            break
        if res is None:
            raise RuntimeError("render_view: no result from renderer")

        alpha = res["alpha_view"].detach().cpu().numpy().transpose(1, 2, 0)
        pred = res["color"].detach().cpu().numpy().transpose(1, 2, 0)
        pred = self.tone_mapper.hdr_to_ldr(pred)
        color_uint8 = (np.concatenate([pred, alpha], axis=-1) * 255.0).astype(np.uint8)

        color_pil = Image.fromarray(color_uint8).resize(
            (resolution, resolution),
            Image.Resampling.LANCZOS,
        )

        base_color = res["base_color"].detach().cpu().numpy().transpose(1, 2, 0)
        metallic = res["metallic"].detach().cpu().numpy().transpose(1, 2, 0)
        roughness = res["roughness"].detach().cpu().numpy().transpose(1, 2, 0)
        shadow = res["shadow"].detach().cpu().numpy().transpose(1, 2, 0)

        return {
            "color": color_pil,
            "base_color": self._to_pil_with_alpha(base_color, alpha),
            "metallic": self._to_pil_with_alpha(metallic, alpha),
            "roughness": self._to_pil_with_alpha(roughness, alpha),
            "shadow": self._to_pil_with_alpha(shadow, alpha),
        }

    @torch.no_grad()
    def render_relight_color_numpy(
        self,
        hs: SparseTensor,
        rfs: torch.Tensor,
        hdri_cond: torch.Tensor,
        yaw_deg: float,
        pitch_deg: float,
        fov_deg: float = 40.0,
        radius: float = 2.0,
        internal_resolution: int = 512,
        bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        clip_near: float = 0.05,
        clip_far: float = 32.0,
    ) -> np.ndarray:
        """Neural relighting: RGB uint8 (H, W, 3) composited over a solid background.

        Expects cached ``hs``, ``rfs`` from ``decoder_pbr_feats`` and ``hdri_cond`` from
        ``encode_hdri`` so interactive viewers can skip decoder / HDRI encoder each frame.

        ``clip_near`` / ``clip_far`` are the gsplat perspective clip planes in view space.
        Cameras use ``extrinsics_look_at`` at distance ``radius`` from the origin, so scene
        depths are on the order of ``radius``. The legacy defaults ``near=1, far=3`` only fit
        ``radius`` around ~2; larger orbit radii need a larger ``clip_far`` or the object
        disappears past the far plane.
        """
        res_i = int(internal_resolution)
        cache_key = (res_i, tuple(bg_color), float(clip_near), float(clip_far))
        if getattr(self, "_relight_render_key", None) != cache_key:
            self.setup_renderer(
                resolution=res_i,
                near=clip_near,
                far=clip_far,
                bg_color=bg_color,
                ssaa=1,
            )
            self._relight_render_key = cache_key

        extr, intr = self.generate_camera(yaw_deg, pitch_deg, radius, fov_deg)
        reps = self.get_reps(hs, rfs, hdri_cond, extr[None, ...])
        res = None
        for rep in reps:
            res = self.renderer.render(
                rep,
                extr,
                intr,
                opt=edict(neural_basis=self.models["neural_basis"]),
            )
            break
        if res is None:
            raise RuntimeError("render_relight_color_numpy: empty reps")

        alpha = res["alpha_view"].detach().cpu().numpy().transpose(1, 2, 0)
        pred = res["color"].detach().cpu().numpy().transpose(1, 2, 0)
        pred = self.tone_mapper.hdr_to_ldr(pred)
        bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
        frame = pred * alpha + (1.0 - alpha) * bg
        return (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)

    @torch.no_grad()
    def render_camera_path_video(
        self,
        slat: SparseTensor,
        hdri_np: np.ndarray,
        num_views: int,
        fov: float = 40.0,
        radius: float = 2.0,
        hdri_rot_deg: float = 0.0,
        full_video: bool = False,
        shadow_video: bool = False,
        bg_color: Tuple[float, float, float] = (0, 0, 0),
        verbose: bool = True,
    ) -> List[np.ndarray]:
        """Render a video along the default spiral camera path."""
        hdri_cond = self.encode_hdri(hdri_np, hdri_rot_deg)
        hs, rfs = self.decoder_pbr_feats(slat)
        extrinsics, intrinsics = self.generate_spiral_cameras(num_views, radius, fov)
        num_views = len(extrinsics)
        hdri_conds = hdri_cond.repeat(num_views, 1, 1)
        frames: List[np.ndarray] = []

        iterator: Any = enumerate(zip(extrinsics, intrinsics, hdri_conds))
        if verbose:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=num_views, desc="Rendering camera path")

        bg = np.array(bg_color)
        for _, (extr, intr, hc) in iterator:
            reps = self.get_reps(hs, rfs, hc[None, ...], extr[None, ...])
            for rep in reps:
                res = self.renderer.render(rep, extr, intr, opt=edict(neural_basis=self.models["neural_basis"]))
                color = res["color"].detach().cpu().numpy().transpose(1, 2, 0)
                alpha = res["alpha_view"].detach().cpu().numpy().transpose(1, 2, 0)
                color = self.tone_mapper.hdr_to_ldr(color)
                if not full_video:
                    frame = color * alpha + (1 - alpha) * bg
                    frames.append((frame * 255).astype(np.uint8))
                else:
                    base_color = res["base_color"].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * bg[:1]
                    metallic = res["metallic"].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * bg[:1]
                    roughness = res["roughness"].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * bg[:1]
                    shadow = res["shadow"].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * bg
                    parts = [
                        color * alpha + (1 - alpha) * bg,
                        base_color,
                        metallic.repeat(3, axis=-1),
                        roughness.repeat(3, axis=-1),
                    ]
                    if shadow_video:
                        parts.append(shadow)
                    frames.append((np.concatenate(parts, axis=1) * 255).astype(np.uint8))
                break
        return frames

    @torch.no_grad()
    def render_hdri_rotation_video(
        self,
        slat: SparseTensor,
        hdri_np: np.ndarray,
        num_frames: int,
        yaw_deg: float,
        pitch_deg: float,
        fov: float = 40.0,
        radius: float = 2.0,
        full_video: bool = False,
        shadow_video: bool = False,
        bg_color: Tuple[float, float, float] = (0, 0, 0),
        verbose: bool = True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Render a video with fixed camera and rotating HDRI illumination."""
        hdri_rots_rad = 2 * np.pi * np.arange(num_frames) / num_frames
        hdri_conds = [self.encode_hdri(hdri_np, np.rad2deg(r)) for r in hdri_rots_rad]
        hdri_conds = torch.stack(hdri_conds, dim=0)

        hdri_roll_frames: List[np.ndarray] = []
        for r in hdri_rots_rad:
            ldr = self.hdri_processor.rotate_hdri_and_get_cond(hdri_np, r, self.tone_mapper)
            ldr = (np.clip(ldr, 0, 1) * 255).astype(np.uint8)
            if ldr.shape[0] == 3:
                ldr = np.transpose(ldr, (1, 2, 0))
            hdri_roll_frames.append(ldr)

        extr, intr = self.generate_camera(yaw_deg, pitch_deg, radius, fov)
        extr = extr.to(self.device)
        intr = intr.to(self.device)
        hs, rfs = self.decoder_pbr_feats(slat)
        render_frames: List[np.ndarray] = []

        iterator: Any = range(num_frames)
        if verbose:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Rendering HDRI rotation")

        bg = np.array(bg_color)
        for i in iterator:
            hc = hdri_conds[i]
            reps = self.get_reps(hs, rfs, hc, extr[None, ...])
            for rep in reps:
                res = self.renderer.render(rep, extr, intr, opt=edict(neural_basis=self.models["neural_basis"]))
                color = res["color"].detach().cpu().numpy().transpose(1, 2, 0)
                alpha = res["alpha_view"].detach().cpu().numpy().transpose(1, 2, 0)
                color = self.tone_mapper.hdr_to_ldr(color)
                if not full_video:
                    frame = color * alpha + (1 - alpha) * bg
                    render_frames.append((frame * 255).astype(np.uint8))
                else:
                    base_color = res["base_color"].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * bg[:1]
                    metallic = res["metallic"].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * bg[:1]
                    roughness = res["roughness"].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * bg[:1]
                    shadow = res["shadow"].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * bg
                    parts = [
                        color * alpha + (1 - alpha) * bg,
                        base_color,
                        metallic.repeat(3, axis=-1),
                        roughness.repeat(3, axis=-1),
                    ]
                    if shadow_video:
                        parts.append(shadow)
                    render_frames.append((np.concatenate(parts, axis=1) * 255).astype(np.uint8))
                break
        return hdri_roll_frames, render_frames

    def export_glb_from_slat(
        self,
        slat: SparseTensor,
        hdri_np: np.ndarray,
        hdri_rot_deg: float = 0.0,
        base_mesh: Optional[trimesh.Trimesh] = None,
        simplify: float = 0.95,
        texture_size: int = 2048,
        fill_holes: bool = True,
    ) -> trimesh.Trimesh:
        """Export a textured PBR GLB, preferring an externally provided base mesh."""
        if base_mesh is None and "slat_decoder_mesh" in self.models:
            mesh_out = self.models["slat_decoder_mesh"](slat)
            base_mesh = mesh_out[0] if isinstance(mesh_out, (list, tuple)) else mesh_out
        if base_mesh is None:
            raise ValueError(
                "export_glb_from_slat requires `base_mesh` or pipeline model `slat_decoder_mesh`"
            )

        hdri_cond = self.encode_hdri(hdri_np, hdri_rot_deg)
        hs, rfs = self.decoder_pbr_feats(slat)
        return render_utils_rl.to_glb(
            self.models["renderer"],
            hs,
            rfs,
            hdri_cond,
            self.models["neural_basis"],
            base_mesh,
            tone_mapper=self.tone_mapper,
            simplify=simplify,
            fill_holes=fill_holes,
            texture_size=texture_size,
        )

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal["stochastic", "multidiffusion"] = "stochastic",
    ) -> Iterator[None]:
        """Temporarily inject multi-image conditioning behavior into a sampler."""
        sampler = getattr(self, sampler_name)
        setattr(sampler, "_old_inference_model", sampler._inference_model)

        if mode == "stochastic":
            if num_images > num_steps:
                print(
                    f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m"
                )

            cond_indices = (np.arange(num_steps) % num_images).tolist()

            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx : cond_idx + 1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)

        elif mode == "multidiffusion":
            from .samplers import FlowEulerSampler

            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = [
                        FlowEulerSampler._inference_model(self, model, x_t, t, cond[i : i + 1], **kwargs)
                        for i in range(len(cond))
                    ]
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred

                preds = [
                    FlowEulerSampler._inference_model(self, model, x_t, t, cond[i : i + 1], **kwargs)
                    for i in range(len(cond))
                ]
                return sum(preds) / len(preds)

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))
        yield
        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, "_old_inference_model")

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        mesh: trimesh.Trimesh,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
        preprocess_image: bool = True,
        mode: Literal["stochastic", "multidiffusion"] = "stochastic",
    ) -> Any:
        """Run multi-image conditioning with an externally provided geometry mesh."""
        del num_samples, sparse_structure_sampler_params, formats

        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond["neg_cond"] = cond["neg_cond"][:1]
        torch.manual_seed(seed)
        coords = self.shape_to_coords(mesh)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get("steps")
        with self.inject_sampler_multi_image("slat_sampler", len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        hs, rfs = self.decoder_pbr_feats(slat)
        return hs, rfs
