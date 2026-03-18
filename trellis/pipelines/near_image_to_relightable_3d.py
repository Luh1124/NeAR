from typing import *
from contextlib import contextmanager
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pyexr
from torchvision import transforms
from PIL import Image
from easydict import EasyDict as edict
import trimesh
import open3d as o3d
import rembg
from simple_ocio import ToneMapper
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from ..datasets.hdri_processer import HDRI_Preprocessor
from ..utils.tool import imageSuperNet
from ..utils import render_utils_rl


class NeARImageToRelightable3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sr_model_path: str = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.slat_sampler = slat_sampler
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)
        self.hdri_processor = HDRI_Preprocessor(envmap_h=512, envmap_w=1024)
        self.sr_session = None
        # self.sr_session = imageSuperNet(sr_model_path) if sr_model_path is not None else None
        self.renderer = None
        # self.setup_renderer()
        self.tone_mapper = None
        # self.tone_mapper = ToneMapper()
        # self.setup_tone_mapper('AgX')

    def setup_renderer(self, resolution: int = 512, near: float = 1, far: float = 3, bg_color: Tuple[float, float, float] = (0, 0, 0), ssaa: int = 1):
        """
        Setup the renderer.
        """
        self.renderer = render_utils_rl.get_renderer(
            resolution=resolution,
            near=near,
            far=far,
            bg_color=bg_color,
            ssaa=ssaa,
        )

    def setup_tone_mapper(self, view: str = 'AgX'):
        """
        Setup the tone mapper.
        """
        self.tone_mapper = ToneMapper()
        self.tone_mapper.view = view

    @staticmethod
    def from_pretrained(path: str) -> "NeARImageToRelightable3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(NeARImageToRelightable3DPipeline, NeARImageToRelightable3DPipeline).from_pretrained(path)
        new_pipeline = NeARImageToRelightable3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        sr_path = args['sr_model']['model_path']
        if not os.path.isabs(sr_path):
            sr_path = os.path.join(path, sr_path)
        new_pipeline.sr_session = imageSuperNet(sr_path)

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']
        new_pipeline._init_image_cond_model(args['image_cond_model'])
        new_pipeline.hdri_processor = HDRI_Preprocessor(envmap_h=512, envmap_w=1024)
        new_pipeline.setup_renderer()
        new_pipeline.setup_tone_mapper('AgX')

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True).eval()
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output
    
    def process_hdri(self, hdri, hdri_rot=0):
        hdri = cv2.resize(hdri, (1024, 512), interpolation=cv2.INTER_NEAREST)
        hdri = torch.from_numpy(hdri)
        hdri_rot = [0, 0, hdri_rot * np.pi / 180]
        envir_map_ldr, envir_map_hdr, envir_map_perceptual, envir_map_hdr_raw, view_dirs_world = self.hdri_processor.preprcess_envir_map(
            hdri, hdri_rot)
        hdri_cond = torch.cat([envir_map_ldr, envir_map_hdr, view_dirs_world], dim=0).float()
        return hdri_cond

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
    
    def decoder_pbr_feats(self, slat: sp.SparseTensor) -> Tuple[sp.SparseTensor, torch.Tensor]:
        """
        Decode the structured latent to pbr feats.
        """
        return self.models['decoder'](slat)

    def render_gs(self, slat: sp.SparseTensor, reg_feats: torch.Tensor, hdri_cond: torch.Tensor, extrinsics: torch.Tensor):
        """
        Render the gs.
        """
        return self.models['renderer'](slat, reg_feats, hdri_cond, extrinsics)
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
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
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    @torch.no_grad()
    def shape_to_coords(self, mesh: trimesh.Trimesh) -> torch.Tensor:
        """
        Convert the shape to coordinates.
        """
        transform_y_to_z_up = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        mesh = mesh.apply_transform(transform_y_to_z_up) # rotate to z up, -Y direction
        
        o3d_mesh = o3d.geometry.TriangleMesh()
        vertices = np.clip(np.asarray(mesh.vertices * 0.5), -0.5 + 1e-6, 0.5 - 1e-6)
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(o3d_mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
        vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        vertices = vertices + 0.5
        indices = torch.from_numpy(vertices).to(torch.int32)
        batch_col = torch.zeros(indices.shape[0], 1, dtype=torch.int32)
        coords = torch.cat([batch_col, indices], dim=1).to(self.device)
        return coords

    @torch.no_grad()
    def run_with_shape(
        self,
        image: Image.Image,
        mesh: trimesh.Trimesh,
        seed: int = 42,
        slat_sampler_params: dict = {},
        preprocess_image: bool = True,
    ) -> Any:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            seed (int): The random seed.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        coords = self.shape_to_coords(mesh)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return slat

    @torch.no_grad()
    def run_with_coords(
        self,
        image_list: List[Image.Image],
        coords: torch.Tensor,
        seed: int = 42,
        slat_sampler_params: dict = {},
    ) -> Any:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            return_mesh (bool): Whether to return the mesh.
            preprocess_image (bool): Whether to preprocess the image.
        """
        cond = self.get_cond(image_list)
        torch.manual_seed(seed)
        coords = coords.to(self.device)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return slat 

    @torch.no_grad()
    def encode_hdri(self, hdri_cond, hdri_rot=0):
        hdri_cond = self.process_hdri(hdri_cond, hdri_rot)
        return self.models['hdri_encoder'](hdri_cond[None].to(self.device))

    @torch.no_grad()
    def get_reps(
        self,
        hs: sp.SparseTensor,
        rfs: torch.Tensor,
        hdri_cond: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> Any:
        reps = self.models['renderer'](hs, rfs, hdri_cond, extrinsics)
        return reps

    # --------------- SLaT relighting API (load + render view/video, export GLB) ---------------

    def load_slat(self, slat_path: str) -> sp.SparseTensor:
        """Load SLaT from .npz (feats, coords). coords may be (N,3) or (N,4) with batch dim."""
        loaded = np.load(slat_path, allow_pickle=True)
        feats = torch.from_numpy(loaded['feats']).to(torch.float32)
        coords = torch.from_numpy(loaded['coords']).to(torch.int32)
        if coords.shape[-1] == 3:
            coords = torch.cat([
                torch.zeros((coords.shape[0], 1), device=coords.device, dtype=coords.dtype),
                coords
            ], dim=-1)
        return sp.SparseTensor(feats, coords).to(self.device)

    def load_hdri(self, hdri_path: str) -> np.ndarray:
        """Load HDRI EXR; returns float array (H, W, 3). Requires pyexr."""
        return pyexr.read(hdri_path)[..., :3]

    def generate_camera(self, yaw_deg: float, pitch_deg: float, radius: float, fov: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single view: (extrinsic, intrinsic). Angles in degrees."""
        yaw_rad = np.deg2rad(float(yaw_deg))
        pitch_rad = np.deg2rad(float(pitch_deg))
        extr, intr = render_utils_rl.yaw_pitch_r_fov_to_extrinsics_intrinsics([yaw_rad], [pitch_rad], radius, fov)
        return extr[0].to(self.device), intr[0].to(self.device)

    def generate_spiral_cameras(self, num_views: int, radius: float = 2.0, fov: float = 40.0) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Spiral camera path for video. Returns (extrinsics, intrinsics)."""
        extrinsics, intrinsics = render_utils_rl.generate_cameras_spiral(num_views, r=radius, fov=fov)
        extrinsics = [e.to(self.device) for e in extrinsics]
        intrinsics = [i.to(self.device) for i in intrinsics]
        return extrinsics, intrinsics

    @torch.no_grad()
    def render_view(
        self,
        slat: sp.SparseTensor,
        hdri_np: np.ndarray,
        yaw_deg: float,
        pitch_deg: float,
        fov: float = 40.0,
        radius: float = 2.0,
        hdri_rot_deg: float = 0.0,
        super_resolve: bool = False,
        resolution: int = 1024,
    ) -> Dict[str, Image.Image]:
        """
        Render a single view. Returns dict with keys: color, base_color, metallic, roughness, shadow (PIL Images).
        """
        hdri_cond = self.encode_hdri(hdri_np, hdri_rot_deg)
        hs, rfs = self.decoder_pbr_feats(slat)
        extr, intr = self.generate_camera(yaw_deg, pitch_deg, radius, fov)
        reps = self.get_reps(hs, rfs, hdri_cond, extr[None, ...])
        res = None
        for rep in reps:
            res = self.renderer.render(rep, extr, intr, opt=edict(neural_basis=self.models['neural_basis']))
            break
        if res is None:
            raise RuntimeError("render_view: no result from renderer")
        alpha = res['alpha_view'].detach().cpu().numpy().transpose(1, 2, 0)
        pred = res['color'].detach().cpu().numpy().transpose(1, 2, 0)
        pred = self.tone_mapper.hdr_to_ldr(pred)
        color_uint8 = (np.concatenate([pred, alpha], axis=-1) * 255.0).astype(np.uint8)
        if super_resolve and getattr(self, 'sr_session', None) is not None:
            color_pil = self.sr_session(Image.fromarray(color_uint8))
        else:
            color_pil = Image.fromarray(color_uint8).resize((resolution, resolution), Image.Resampling.LANCZOS)
        if not isinstance(color_pil, Image.Image):
            color_pil = Image.fromarray((np.clip(color_pil, 0, 255)).astype(np.uint8))
        base_color = res['base_color'].detach().cpu().numpy().transpose(1, 2, 0)
        metallic = res['metallic'].detach().cpu().numpy().transpose(1, 2, 0)
        roughness = res['roughness'].detach().cpu().numpy().transpose(1, 2, 0)
        shadow = res['shadow'].detach().cpu().numpy().transpose(1, 2, 0)
        def to_pil(x: np.ndarray, alpha_ch: np.ndarray) -> Image.Image:
            x = np.concatenate([x.repeat(3, axis=-1) if x.shape[-1] == 1 else x, alpha_ch], axis=-1)
            return Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))
        return {
            'color': color_pil,
            'base_color': to_pil(base_color, alpha),
            'metallic': to_pil(metallic, alpha),
            'roughness': to_pil(roughness, alpha),
            'shadow': to_pil(shadow, alpha),
        }

    @torch.no_grad()
    def render_camera_path_video(
        self,
        slat: sp.SparseTensor,
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
        """
        Render frames along spiral camera path. Returns list of (H,W,3) uint8 frames.
        If full_video: each frame is horizontal stack [color|base_color|metallic|roughness|shadow].
        """
        hdri_cond = self.encode_hdri(hdri_np, hdri_rot_deg)
        hs, rfs = self.decoder_pbr_feats(slat)
        extrinsics, intrinsics = self.generate_spiral_cameras(num_views, radius, fov)
        num_views = len(extrinsics)
        hdri_conds = hdri_cond.repeat(num_views, 1, 1)
        frames = []
        it = enumerate(zip(extrinsics, intrinsics, hdri_conds))
        if verbose:
            from tqdm import tqdm
            it = tqdm(it, total=num_views, desc="Rendering camera path")
        for i, (extr, intr, hc) in it:
            reps = self.get_reps(hs, rfs, hc[None, ...], extr[None, ...])
            for rep in reps:
                res = self.renderer.render(rep, extr, intr, opt=edict(neural_basis=self.models['neural_basis']))
                color = res['color'].detach().cpu().numpy().transpose(1, 2, 0)
                alpha = res['alpha_view'].detach().cpu().numpy().transpose(1, 2, 0)
                color = self.tone_mapper.hdr_to_ldr(color)
                if not full_video:
                    frame = (color * alpha + (1 - alpha) * np.array(bg_color))
                    frames.append((frame * 255).astype(np.uint8))
                else:
                    bc = res['base_color'].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * np.array(bg_color)[:1]
                    met = res['metallic'].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * np.array(bg_color)[:1]
                    rough = res['roughness'].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * np.array(bg_color)[:1]
                    shad = res['shadow'].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * np.array(bg_color)
                    parts = [color * alpha + (1 - alpha) * np.array(bg_color), bc, met.repeat(3, axis=-1), rough.repeat(3, axis=-1)]
                    if shadow_video:
                        parts.append(shad)
                    frame = np.concatenate(parts, axis=1)
                    frames.append((frame * 255).astype(np.uint8))
        return frames

    @torch.no_grad()
    def render_hdri_rotation_video(
        self,
        slat: sp.SparseTensor,
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
        """
        HDRI rotates; camera fixed. Returns (hdri_roll_frames, render_frames).
        """
        hdri_rots_rad = 2 * np.pi * np.arange(num_frames) / num_frames
        hdri_conds = []
        for r in hdri_rots_rad:
            hdri_conds.append(self.encode_hdri(hdri_np, np.rad2deg(r)))
        hdri_conds = torch.stack([c for c in hdri_conds], dim=0)
        hdri_roll_frames = []
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
        render_frames = []
        it = range(num_frames)
        if verbose:
            from tqdm import tqdm
            it = tqdm(it, desc="Rendering HDRI rotation")
        for i in it:
            hc = hdri_conds[i]
            reps = self.get_reps(hs, rfs, hc, extr[None, ...])
            for rep in reps:
                res = self.renderer.render(rep, extr, intr, opt=edict(neural_basis=self.models['neural_basis']))
                color = res['color'].detach().cpu().numpy().transpose(1, 2, 0)
                alpha = res['alpha_view'].detach().cpu().numpy().transpose(1, 2, 0)
                color = self.tone_mapper.hdr_to_ldr(color)
                if not full_video:
                    frame = (color * alpha + (1 - alpha) * np.array(bg_color))
                    render_frames.append((frame * 255).astype(np.uint8))
                else:
                    bc = res['base_color'].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * np.array(bg_color)[:1]
                    met = res['metallic'].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * np.array(bg_color)[:1]
                    rough = res['roughness'].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * np.array(bg_color)[:1]
                    shad = res['shadow'].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * np.array(bg_color)
                    parts = [color * alpha + (1 - alpha) * np.array(bg_color), bc, met.repeat(3, axis=-1), rough.repeat(3, axis=-1)]
                    if shadow_video:
                        parts.append(shad)
                    render_frames.append((np.concatenate(parts, axis=1) * 255).astype(np.uint8))
        return hdri_roll_frames, render_frames

    @torch.no_grad()
    def export_glb_from_slat(
        self,
        slat: sp.SparseTensor,
        hdri_np: np.ndarray,
        hdri_rot_deg: float = 0.0,
        base_mesh: Optional[trimesh.Trimesh] = None,
        simplify: float = 0.95,
        texture_size: int = 2048,
        fill_holes: bool = True,
    ) -> trimesh.Trimesh:
        """
        Export PBR GLB mesh. base_mesh required unless pipeline has slat_decoder_mesh.
        Returns trimesh with PBR texture; call .export(path) to write .glb.
        """
        if base_mesh is None and 'slat_decoder_mesh' in self.models:
            mesh_out = self.models['slat_decoder_mesh'](slat)
            base_mesh = mesh_out[0] if isinstance(mesh_out, (list, tuple)) else mesh_out
        if base_mesh is None:
            raise ValueError("export_glb_from_slat requires base_mesh or pipeline model 'slat_decoder_mesh'")
        hdri_cond = self.encode_hdri(hdri_np, hdri_rot_deg)
        hs, rfs = self.decoder_pbr_feats(slat)
        return render_utils_rl.to_glb(
            self.models['renderer'],
            hs, rfs, hdri_cond,
            self.models['neural_basis'],
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
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        mesh: trimesh.Trimesh,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> Any:
        """
        Run the pipeline with multiple images as condition. 几何 mesh 需在外部用 Hunyuan3D 等先生成后传入。
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        coords = self.shape_to_coords(mesh)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        hs, rfs = self.decoder_pbr_feats(slat)
        return hs, rfs

    

    
    
