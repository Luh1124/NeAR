#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torchvision
import math
from easydict import EasyDict as edict
import numpy as np
from ..representations.gaussian import Gaussian_view as Gaussian
from .sh_utils import eval_sh
import torch.nn.functional as F
from easydict import EasyDict as edict

from gsplat.rendering import rasterization as gsplat_rasterization
# from .normal_utils import compute_normals
from .graphics_utils import depths_to_points

def batched_neural_basis(model, normals, point_features: torch.Tensor, batch_size: int) -> torch.Tensor:  
    """  
    分批运行 NeuralBasis 模型以节省显存。  
    
    Args:  
        model: NeuralBasis 模型实例。  
        dirs: 方向向量，形状为 [N, 3]，N是高斯点总数。  
        point_features: 点特征，形状为 [N, C, feature_dim]，C通常是3(RGB)。  
        batch_size: 每个批次处理的点数量。  
        
    Returns:  
        计算出的颜色，形状为 [N, C]。  
    """  
    num_points = normals.shape[0]  
    if num_points == 0:  
        return torch.empty(0, point_features.shape[1], device=normals.device)  

    results = []  
    # 使用 torch.split 进行高效切分  
    for normals_batch, features_batch in zip(  
        torch.split(normals, batch_size, dim=0),  
        torch.split(point_features, batch_size, dim=0)  
    ):  
        # 调用模型前向传播  
        colors_batch = model(features_batch, normals_batch)  
        results.append(colors_batch)  
    
    # 将所有批次的结果拼接起来  
    return torch.cat(results, dim=0)  

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap  shape: (H, W)
    """
    points = depths_to_points(view, depth).reshape(*depth.shape, 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output.permute(2, 0, 1)

def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret


def compute_depth_normal(viewpoint_camera, rendered_depth, rendered_alpha):
    rendered_normal_from_depth = depth_to_normal(viewpoint_camera, rendered_depth.squeeze()) 
    rendered_normal_from_depth = rendered_normal_from_depth.permute(1, 2, 0)  # 调整为 H x W x C
    view_transform = viewpoint_camera.world_view_transform[:3, :3].unsqueeze(0)  # H x W x 3 x 3
    rendered_alpha = rendered_alpha.unsqueeze(-1)  # H x W x 1
    rendered_normal_from_depth = rendered_normal_from_depth @ view_transform * rendered_alpha.detach()  # H x W x 3
    rendered_normal_from_depth = rendered_normal_from_depth[0].permute(2, 0, 1)  # 变为 3 x H x W
    rendered_normal_from_depth[1:3] = - rendered_normal_from_depth[1:3] # y,z axis flip

    return rendered_normal_from_depth

def render_gsplat(viewpoint_camera, pc : Gaussian, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, neural_basis: torch.nn.Module = torch.nn.Identity()):
    """
    使用高斯泼溅(Gaussian Splatting)技术渲染3D场景。
    
    参数:
        viewpoint_camera: 相机视角对象，包含相机位置、视角变换等参数
        pc: Gaussian_view对象，包含3D高斯点的位置、颜色、材质等属性
        pipe: 渲染管线配置对象
        bg_color: 背景颜色张量(必须位于GPU上)
        scaling_modifier: 缩放修正系数，默认为1.0
        override_color: 可选参数，用于覆盖默认颜色计算
    
    返回:
        edict对象，包含以下渲染结果:
        - render_color: 最终渲染颜色(受阴影影响)
        - render_base_color: 基础颜色
        - render_metallic: 金属度
        - render_roughness: 粗糙度
        - render_shadow: 阴影
        - render_depth: 深度图
        - render_alphas: 透明度
        - info: 其他渲染信息
    
    注意:
        背景张量(bg_color)必须位于GPU上才能正常工作
        支持基于SH(球谐函数)的颜色计算和预计算颜色两种模式


    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    # Set up rasterization configuration for gaussian
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    metallic = pc.get_metallic
    roughness = pc.get_roughness
    pbr1 = pc.get_pbr1
    hdri1 = pc.get_hdri1
    hdri2 = pc.get_hdri2

    # Set up rasterization configuration for gaussian_view
    scales_view = pc.get_scaling_view
    rgb_features = pc.get_rgb

    shadow = pc.get_shadow
    # brightness = pc.get_brightness

    shs_base_color = pc.get_base_color.transpose(1, 2).view(-1, 3, (0+1)**2)
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_base_color.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb_base_color = eval_sh(0, shs_base_color, dir_pp_normalized)
    base_color_precomp = torch.clamp_min(sh2rgb_base_color + 0.5, 0.0)

    global_normal = pc.get_normal(viewpoint_camera)
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3] 

    global_normal_view = pc.get_normal_view(viewpoint_camera)
    local_normal_view = global_normal_view @ viewpoint_camera.world_view_transform[:3,:3] 

    rgb_features_view = batched_neural_basis(neural_basis, global_normal_view, rgb_features, 10000)
    rgb = rgb_features_view[:, :3]
    brightness = rgb_features_view[:, 3:4]
    nush1 = rgb_features_view[:, 4:5]

    # cat the colors with metallic, roughness and shadow
    colors = torch.cat([
        base_color_precomp,
        metallic,
        roughness,
        pbr1,
        local_normal,
    ], dim=-1)

    colors_view = torch.cat([
        rgb,
        shadow,
        brightness,
        nush1,
        hdri1,
        hdri2,
        local_normal_view,
    ], dim=-1)

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    render_colors, render_alphas, info = gsplat_rasterization(
        means = means3D, # [N, 3]
        quats = rotations, # [N, 4]
        scales = scales, # [N, 3]
        opacities = opacity.squeeze(-1), # [N,]
        colors = colors, # [N, 8] base_color + metallic + roughness + normal+1(depth)
        viewmats = viewmat[None],
        Ks = K[None],
        backgrounds=bg_color[:9][None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        near_plane= viewpoint_camera.znear,
        far_plane= viewpoint_camera.zfar,
        distributed = False, # if True, use distributed rendering
        render_mode="RGB+ED",
        rasterize_mode="antialiased",
        packed = False,
    )

    render_colors_view, render_alphas_view, info_view = gsplat_rasterization(
        means = means3D, # [N, 3]
        quats = rotations, # [N, 4]
        scales = scales_view, # [N, 3]
        opacities = opacity.squeeze(-1), # [N,]
        colors = colors_view, # [N, 8] RGB + shadow + normal + 1(depth)
        viewmats = viewmat[None],
        Ks = K[None],
        backgrounds=bg_color[:11][None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        near_plane= viewpoint_camera.znear,
        far_plane= viewpoint_camera.zfar,
        distributed = False, # if True, use distributed rendering
        render_mode="RGB+ED",
        rasterize_mode="antialiased",
        packed = False,
    )

    rendered_image = render_colors[0].permute(2, 0, 1)
    rendered_alpha = render_alphas[0].permute(2, 0, 1)
    rendered_image_view = render_colors_view[0].permute(2, 0, 1)
    rendered_alpha_view = render_alphas_view[0].permute(2, 0, 1)

    rendered_base_color = rendered_image[:3, :, :]
    rendered_metallic = rendered_image[3:4, :, :]
    rendered_roughness = rendered_image[4:5, :, :]
    rendered_pbr1 = rendered_image[5:6, :, :]
    rendered_normal = rendered_image[6:9, :, :]
    rendered_depth = rendered_image[-1:, :, :]

    rendered_final_color = rendered_image_view[:3, :, :]
    rendered_shadow = rendered_image_view[3:4, :, :]
    rendered_brightness = rendered_image_view[4:5, :, :]
    rendered_nush1 = rendered_image_view[5:6, :, :]
    rendered_hdri1 = rendered_image_view[6:7, :, :]
    rendered_hdri2 = rendered_image_view[7:8, :, :]
    rendered_normal_view = rendered_image_view[8:11, :, :]
    rendered_depth_view = rendered_image_view[-1:, :, :]

    rendered_normal[1:3] = - rendered_normal[1:3] # y,z axis flip
    rendered_normal_view[1:3] = - rendered_normal_view[1:3] # y,z axis flip

    rendered_normal_from_depth = compute_depth_normal(viewpoint_camera, rendered_depth, rendered_alpha)
    rendered_normal_from_depth_view = compute_depth_normal(viewpoint_camera, rendered_depth_view, rendered_alpha_view)

    return edict({
        "render": rendered_final_color,
        "render_alpha": rendered_alpha,
        "render_base_color": rendered_base_color,
        "render_metallic": rendered_metallic,
        "render_roughness": rendered_roughness,
        "render_shadow": rendered_shadow,
        "render_pbr1": rendered_pbr1,
        "render_hdri1": rendered_hdri1,
        "render_hdri2": rendered_hdri2,
        "render_nush1": rendered_nush1,
        "render_depth": rendered_depth,
        "render_normal": rendered_normal,
        "render_normal_from_depth": rendered_normal_from_depth,
        "render_shadow_view": rendered_shadow,
        "render_brightness": rendered_brightness,
        "render_alpha_view": rendered_alpha_view,
        "render_depth_view": rendered_depth_view,
        "render_normal_view": rendered_normal_view,
        "render_normal_from_depth_view": rendered_normal_from_depth_view,
        "info": info
        })

def render_gsplat_pbr(viewpoint_camera, pc : Gaussian, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, neural_basis: None = None):
    """
    使用高斯泼溅(Gaussian Splatting)技术渲染3D PBR场景。
    
    参数:
        viewpoint_camera: 相机视角对象，包含相机位置、视角变换等参数
        pc: Gaussian_pbr对象，包含3D高斯点的位置、颜色、材质等属性
        pipe: 渲染管线配置对象
        bg_color: 背景颜色张量(必须位于GPU上)
        scaling_modifier: 缩放修正系数，默认为1.0
        override_color: 可选参数，用于覆盖默认颜色计算
    
    返回:
        edict对象，包含以下渲染结果:
        - render_base_color: 基础颜色
        - render_metallic: 金属度
        - render_roughness: 粗糙度
        - render_pbr1: PBR1
        - render_normal: 法线
        - render_normal_from_depth: 深度法线
        - render_alphas: 透明度
        - info: 其他渲染信息
    
    注意:
        背景张量(bg_color)必须位于GPU上才能正常工作
        支持基于SH(球谐函数)的颜色计算和预计算颜色两种模式


    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    # Set up rasterization configuration for gaussian
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    metallic = pc.get_metallic
    roughness = pc.get_roughness
    pbr1 = pc.get_pbr1

    shs_base_color = pc.get_base_color.transpose(1, 2).view(-1, 3, (0+1)**2)
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_base_color.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb_base_color = eval_sh(0, shs_base_color, dir_pp_normalized)
    base_color_precomp = torch.clamp_min(sh2rgb_base_color + 0.5, 0.0)

    global_normal = pc.get_normal(viewpoint_camera)
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3] 

    # cat the colors with metallic, roughness and shadow
    colors = torch.cat([
        base_color_precomp,
        metallic,
        roughness,
        pbr1,
        local_normal,
    ], dim=-1)

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    render_colors, render_alphas, info = gsplat_rasterization(
        means = means3D, # [N, 3]
        quats = rotations, # [N, 4]
        scales = scales, # [N, 3]
        opacities = opacity.squeeze(-1), # [N,]
        colors = colors, # [N, 8] base_color + metallic + roughness + normal+1(depth)
        viewmats = viewmat[None],
        Ks = K[None],
        backgrounds=bg_color[:9][None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        near_plane= viewpoint_camera.znear,
        far_plane= viewpoint_camera.zfar,
        distributed = False, # if True, use distributed rendering
        render_mode="RGB+ED",
        rasterize_mode="antialiased",
        packed = False,
    )

    rendered_image = render_colors[0].permute(2, 0, 1)
    rendered_alpha = render_alphas[0].permute(2, 0, 1)

    rendered_base_color = rendered_image[:3, :, :]
    rendered_metallic = rendered_image[3:4, :, :]
    rendered_roughness = rendered_image[4:5, :, :]
    rendered_pbr1 = rendered_image[5:6, :, :]
    rendered_normal = rendered_image[6:9, :, :]
    rendered_depth = rendered_image[-1:, :, :]

    rendered_normal[1:3] = - rendered_normal[1:3] # y,z axis flip

    rendered_normal_from_depth = compute_depth_normal(viewpoint_camera, rendered_depth, rendered_alpha)

    return edict({
        "render_alpha": rendered_alpha,
        "render_base_color": rendered_base_color,
        "render_metallic": rendered_metallic,
        "render_roughness": rendered_roughness,
        "render_pbr1": rendered_pbr1,
        "render_depth": rendered_depth,
        "render_normal": rendered_normal,
        "render_normal_from_depth": rendered_normal_from_depth,
        "info": info
        })

class GaussianRenderer:
    """
    Renderer for the Voxel representation.

    Args:
        rendering_options (dict): Rendering options.
    """

    def __init__(self, rendering_options={}) -> None:
        self.rendering_options = edict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1,
            "bg_color": 'random',
            "distributed": False,  # If True, use distributed rendering
        })
        self.rendering_options.update(rendering_options)
        self.pipe = edict({
            "kernel_size": 0.1,
            "convert_SHs_python": True,
            "compute_cov3D_python": False,
            "scale_modifier": 1.0,
            "debug": False,
            "distributed": self.rendering_options["distributed"],  # If True, use distributed rendering
        })
        self.bg_color = None
    
    def render(
            self,
            gaussian: Gaussian,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            colors_overwrite: torch.Tensor = None,
            opt=edict(
                neural_basis=torch.nn.Identity(),
            ),
        ) -> edict:
        """
        Render the gausssian.

        Args:
            gaussian : gaussianmodule
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            colors_overwrite (torch.Tensor): (N, 3) override color

        Returns:
            edict containing:
                color (torch.Tensor): (3, H, W) rendered color image
        """

        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        
        if self.rendering_options["bg_color"] == 'random':
            # self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
            self.bg_color_render = torch.zeros(12, dtype=torch.float32, device="cuda")
            # self.bg_color_render = torch.zeros(10, dtype=torch.float32, device="cuda")
            self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
            if np.random.rand() < 0.5:
                self.bg_color_render += 1
                self.bg_color += 1
        else:
            self.bg_color_render = torch.zeros(12, dtype=torch.float32, device="cuda")
            self.bg_color = torch.tensor(self.rendering_options["bg_color"], dtype=torch.float32, device="cuda")

        view = extrinsics
        perspective = intrinsics_to_projection(intrinsics, near, far)
        camera = torch.inverse(view)[:3, 3]
        focalx = intrinsics[0, 0]
        focaly = intrinsics[1, 1]
        fovx = 2 * torch.atan(0.5 / focalx)
        fovy = 2 * torch.atan(0.5 / focaly)
            
        camera_dict = edict({
            "image_height": resolution * ssaa,
            "image_width": resolution * ssaa,
            "FoVx": fovx,
            "FoVy": fovy,
            "znear": near,
            "zfar": far,
            "world_view_transform": view.T.contiguous(),
            "projection_matrix": perspective.T.contiguous(),
            "full_proj_transform": (perspective @ view).T.contiguous(),
            "camera_center": camera
        })

        # Render
        render_ret = render_gsplat(camera_dict, gaussian, self.pipe, self.bg_color_render, override_color=colors_overwrite, scaling_modifier=self.pipe.scale_modifier, neural_basis=opt.neural_basis)
        # render_ret = render(camera_dict, gausssian, self.pipe, self.bg_color, override_color=colors_overwrite, scaling_modifier=self.pipe.scale_modifier)

        if ssaa > 1:
            render_ret.render = F.interpolate(render_ret.render[None], size=(resolution, resolution), mode='bilinear', align_corners=False, antialias=True).squeeze()
            render_ret.render_base_color = F.interpolate(render_ret.render_base_color[None], size=(resolution, resolution), mode='bilinear', align_corners=False, antialias=True).squeeze()
            render_ret.render_metallic = F.interpolate(render_ret.render_metallic[None], size=(resolution, resolution), mode='bilinear', align_corners=False, antialias=True).squeeze()
            render_ret.render_roughness = F.interpolate(render_ret.render_roughness[None], size=(resolution, resolution), mode='bilinear', align_corners=False, antialias=True).squeeze()
            render_ret.render_shadow = F.interpolate(render_ret.render_shadow[None], size=(resolution, resolution), mode='bilinear', align_corners=False, antialias=True).squeeze()
            render_ret.render_alpha_view = F.interpolate(render_ret.render_alpha_view[None], size=(resolution, resolution), mode='bilinear', align_corners=False, antialias=True).squeeze()
            
        ret = edict({
            'color': render_ret['render'],
            'alpha': render_ret['render_alpha'],
            'base_color': render_ret['render_base_color'],
            'metallic': render_ret['render_metallic'],
            'roughness': render_ret['render_roughness'],
            'shadow': render_ret['render_shadow'],
            'pbr1': render_ret['render_pbr1'],
            'hdri1': render_ret['render_hdri1'],
            'hdri2': render_ret['render_hdri2'],
            'nush1': render_ret['render_nush1'],
            'brightness': render_ret['render_brightness'],
            'alpha_view': render_ret['render_alpha_view'],
            'normal': render_ret['render_normal'],
            'normal_from_depth': render_ret['render_normal_from_depth'],
            'normal_view': render_ret['render_normal_view'],
            'normal_from_depth_view': render_ret['render_normal_from_depth_view'],
            'depth': render_ret['render_depth'],
            'depth_view': render_ret['render_depth_view'],
        })
        return ret

    def debug_save_fig(self, vis_dict):
        for key in vis_dict:
            torchvision.utils.save_image(vis_dict[key][None], f'./debug/debug_{key}.png')
