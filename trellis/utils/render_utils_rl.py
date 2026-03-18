import torch
import numpy as np
from tqdm import tqdm
import utils3d
from PIL import Image
import trimesh
from easydict import EasyDict as edict
from typing import List, Tuple
from ..renderers import GaussianRenderer
from ..representations import Octree, Gaussian, MeshExtractResult
from ..modules import sparse as sp
from .random_utils import sphere_hammersley_sequence
from . import postprocessing_utils

def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda() * r
        extr = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics

def generate_cameras_spiral(num_views: int, r=2, fov=40) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """生成yaw和pitch的螺旋序列"""
    yaws = torch.linspace(0, 2 * np.pi, num_views)
    pitchs = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * np.pi, num_views))
    yaws = yaws.tolist()
    pitchs = pitchs.tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    return extrinsics, intrinsics

def get_renderer(**kwargs):
    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = kwargs.get('resolution', 512)
    renderer.rendering_options.near = kwargs.get('near', 1)
    renderer.rendering_options.far = kwargs.get('far', 3)
    renderer.rendering_options.bg_color = kwargs.get('bg_color', (1, 1, 1))
    renderer.rendering_options.ssaa = kwargs.get('ssaa', 1)
    return renderer

def render_frames(rep_renderer, h, reg_feats, hdri_cond, extrinsics, intrinsics, options={}, colors_overwrite=None, verbose=True, neural_basis=None, tone_mapper=None, return_brm=False, **kwargs):
    renderer = get_renderer(**options)
    rets = {}
    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)),  desc='Rendering', total=len(extrinsics), disable=not verbose):
        rep = rep_renderer(h, reg_feats, hdri_cond, extr[None])
        res = renderer.render(rep[0], extr, intr, colors_overwrite=colors_overwrite, opt=edict(neural_basis=neural_basis))
        if 'color' not in rets: rets['color'] = []
        if 'base_color' not in rets: rets['base_color'] = []
        if 'roughness' not in rets: rets['roughness'] = []
        if 'metallic' not in rets: rets['metallic'] = []
        if 'shadow' not in rets: rets['shadow'] = []
        if 'alpha' not in rets: rets['alpha'] = []
        if 'brm' not in rets: rets['brm'] = []

        alpha = res['alpha_view'].detach().cpu().numpy().transpose(1, 2, 0) # 512 512 1
        base_color = res['base_color'].detach().cpu().numpy().transpose(1, 2, 0) # 512 512 3
        roughness = res['roughness'].detach().cpu().numpy().transpose(1, 2, 0) # 512 512 1
        metallic = res['metallic'].detach().cpu().numpy().transpose(1, 2, 0) # 512 512 1
        shadow = res['shadow'].detach().cpu().numpy().transpose(1, 2, 0) # 512 512 1

        color = res['color'].detach().cpu().numpy().transpose(1, 2, 0) # 512 512 3
        color = tone_mapper.hdr_to_ldr(color) # 512 512 3

        rets['alpha'].append((alpha * 255).astype(np.uint8))
        rets['base_color'].append(((base_color * alpha + (1 - alpha) * options['bg_color'])*255).astype(np.uint8))

        if return_brm:
            rets['brm'].append(((np.concatenate([np.zeros_like(roughness), roughness, metallic], axis=-1) * alpha)*255).astype(np.uint8))
            continue
        else:
            rets['color'].append(((color * alpha + (1 - alpha) * options['bg_color'])*255).astype(np.uint8))
            rets['roughness'].append(((roughness * alpha + (1 - alpha) * options['bg_color'][:1])*255).astype(np.uint8))
            rets['metallic'].append(((metallic * alpha + (1 - alpha) * options['bg_color'][:1])*255).astype(np.uint8))
            rets['shadow'].append(((shadow * alpha + (1 - alpha) * options['bg_color'])*255).astype(np.uint8))

    return edict(rets)

def render_video(render_gs, h, reg_feats, hdri_cond, neural_basis, tone_mapper, resolution=512, bg_color=(1, 1, 1), num_frames=300, r=2, fov=40, **kwargs):
    extrinsics, intrinsics = generate_cameras_spiral(num_frames, r, fov)
    return render_frames(render_gs, h, reg_feats, hdri_cond, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, neural_basis=neural_basis, tone_mapper=tone_mapper, **kwargs)

def render_single_view(render_gs, h, reg_feats, hdri_cond, neural_basis, tone_mapper, yaw, pitch, r=2, fov=40, resolution=512, bg_color=(1, 1, 1), **kwargs):
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics([yaw], [pitch], r, fov)
    return render_frames(render_gs, h, reg_feats, hdri_cond, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, neural_basis=neural_basis, tone_mapper=tone_mapper, **kwargs)


def render_multiview(render_gs, h, reg_feats, hdri_cond, neural_basis, tone_mapper, resolution=512, bg_color=(0, 0, 0), nviews=30, r=2, fov=40, return_brm=True):
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    rets = render_frames(render_gs, h, reg_feats, hdri_cond, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, neural_basis=neural_basis, tone_mapper=tone_mapper, return_brm=return_brm)
    obs = {
        'alpha': rets['alpha'],
        'base_color': rets['base_color'],
        'brm': rets['brm']
    }
    return edict(obs), extrinsics, intrinsics


def to_glb(
        renderer,
        h,
        reg_feats,
        hdri_cond,
        neural_basis,
        mesh, 
        tone_mapper,
        simplify: float = 0.95,
        fill_holes: bool = True,
        fill_holes_max_size: float = 0.04,
        texture_size: int = 1024, 
        debug: bool = False,
        verbose: bool = False,
    ) -> trimesh.Trimesh:

    if isinstance(mesh, trimesh.Trimesh):
        vertices = mesh.vertices
        faces = mesh.faces
    else:
        vertices = mesh.vertices.cpu().detach().numpy()
        faces = mesh.faces.cpu().detach().numpy()

    # mesh postprocess
    vertices, faces = postprocessing_utils.postprocess_mesh(
        vertices, faces,
        simplify=simplify > 0,
        simplify_ratio=simplify,
        fill_holes=fill_holes,
        fill_holes_max_hole_size=fill_holes_max_size,
        fill_holes_max_hole_nbe=int(250 * np.sqrt(1-simplify)),
        fill_holes_resolution=1024,
        fill_holes_num_views=1000,
        debug=debug,
        verbose=verbose,
    )

    vertices, faces, uvs = postprocessing_utils.parametrize_mesh(vertices, faces)

    with torch.inference_mode():
        observations, extrinsics, intrinsics = render_multiview(renderer, h, reg_feats, hdri_cond, neural_basis, tone_mapper, resolution=1024, nviews=100)

    masks = [np.any(obs_alpha > 0, axis=-1) for obs_alpha in observations['alpha']]
    extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
    intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]
    
    base_color_texture = postprocessing_utils.bake_texture(
        vertices, faces, uvs,
        observations['base_color'], masks, extrinsics, intrinsics,
        texture_size=texture_size, mode='opt',
        lambda_tv=0.01,
        verbose=verbose
    )
    brm_texture = postprocessing_utils.bake_texture(
        vertices, faces, uvs,
        observations['brm'], masks, extrinsics, intrinsics,
        texture_size=texture_size, mode='opt',
        lambda_tv=0.01,
        verbose=verbose
    )
    base_color_texture = Image.fromarray(base_color_texture)
    brm_texture = Image.fromarray(brm_texture)

    # Create material with proper trimesh API
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=base_color_texture,
        metallicRoughnessTexture=brm_texture,
    )

    # Create mesh with texture visuals
    visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual)
    return mesh