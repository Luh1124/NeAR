import os
import gradio as gr

# Workaround for Gradio/gradio_client bug: get_type() receives bool (e.g. additionalProperties: true)
# and raises TypeError: argument of type 'bool' is not iterable. Patch to handle non-dict schema.
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

import glob
import json
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Any
from PIL import Image
import shutil
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from easydict import EasyDict as edict
from safetensors.torch import load_file

# 第三方库
import pyexr
import imageio
import utils3d
from simple_ocio import ToneMapper


tone_mapper = ToneMapper()
tone_mapper.view = 'AgX'

# ========================= 配置和常量 =========================
# 环境变量配置（注意：TORCH_CUDA_ARCH_LIST 仅影响「从源码编译」时的目标架构，
# 已安装的 gsplat 等预编译包不会改变。若出现 "no kernel image is available for execution on the device"，
# 说明当前 GPU 架构与 gsplat 编译时不一致，需用本机架构重新安装 gsplat，见下方启动时的 GPU 检测输出。）
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

# TRELLIS相关导入
from trellis import models
from trellis.models import SLatRadianceFieldDecoder, SLatMeshDecoder, SLatGaussianDecoder
from trellis.modules import sparse as sp
from trellis.representations import Gaussian_view as Gaussian
from trellis.datasets.hdri_processer import HDRI_Preprocessor
from trellis.utils import render_utils, render_utils_rl


class imageSuperNet:
    def __init__(self, config) -> None:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path=config.realesrgan_ckpt_path,
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            gpu_id=None,
        )
        self.upsampler = upsampler

    def __call__(self, image):
        output, _ = self.upsampler.enhance(np.array(image))
        output = Image.fromarray(output)
        return output


def get_model_summary(model):
    model_summary = 'Parameters:\n'
    model_summary += '=' * 128 + '\n'
    model_summary += f'{"Name":<{72}}{"Shape":<{32}}{"Type":<{16}}{"Grad"}\n'
    num_params = 0
    num_trainable_params = 0
    for name, param in model.named_parameters():
        model_summary += f'{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n'
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    model_summary += '\n'
    model_summary += f'Number of parameters: {num_params / 1e6:.2f}M\n'
    model_summary += f'Number of trainable parameters: {num_trainable_params / 1e6:.2f}M\n'
    return model_summary

# 解码器字典
DECODER_DICT = {
    "gs": SLatGaussianDecoder,
    "rf": SLatRadianceFieldDecoder,
    "mesh": SLatMeshDecoder,
}

# 全局配置
# 注意：下面所有模型都在「进程启动」时加载，Gradio 要等这些跑完才会绑定端口。
# 所以从执行 python 到能在浏览器打开页面，会花较长时间（主要耗时在 5 个 TRELLIS 模型 + RealESRGAN）。
_models = edict()
_t0 = time.perf_counter()
_models['hdri_encoder'] = models.from_pretrained('../scaffordrelit_weights/rl_even_ckpts/hdri_encoder_4096tokens_fp16').requires_grad_(False)
print(f"  [启动] hdri_encoder 加载完成, +{time.perf_counter() - _t0:.1f}s")
_models['decoder'] = models.from_pretrained('../scaffordrelit_weights/rl_even_ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16').requires_grad_(False)
print(f"  [启动] decoder 加载完成, +{time.perf_counter() - _t0:.1f}s")
_models['renderer'] = models.from_pretrained('../scaffordrelit_weights/rl_even_ckpts/slat_renderder_gs_swin8_B_64l8gs32_fp16').requires_grad_(False)
print(f"  [启动] renderer 加载完成, +{time.perf_counter() - _t0:.1f}s")
_models['neural_basis'] = models.from_pretrained('../scaffordrelit_weights/rl_even_ckpts/neural_basis_3layer_relu_fp16').requires_grad_(False)
print(f"  [启动] neural_basis 加载完成, +{time.perf_counter() - _t0:.1f}s")
_models['slat_decoder_mesh'] = models.from_pretrained('/root/code/3diclight/trelli_ckpts/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16').requires_grad_(False)
print(f"  [启动] slat_decoder_mesh 加载完成, +{time.perf_counter() - _t0:.1f}s")

# model summary
print(get_model_summary(_models['decoder']))
print(get_model_summary(_models['renderer']))
print(get_model_summary(_models['hdri_encoder']))
print(get_model_summary(_models['neural_basis']))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
if device == "cuda":
    cap = torch.cuda.get_device_capability(0)
    arch_str = f"{cap[0]}.{cap[1]}"
    name = torch.cuda.get_device_name(0)
    print(f"  [GPU] {name} , Compute Capability {arch_str}")
    if arch_str != "8.9":
        print(f"  [提示] 当前 GPU 架构为 {arch_str}，与脚本中 TORCH_CUDA_ARCH_LIST=8.9 不一致。")
        print(f"         若运行 gsplat 报错 'no kernel image is available'，请在本机用对应架构重新安装 gsplat，例如：")
        print(f"         TORCH_CUDA_ARCH_LIST=\"{arch_str}\" pip install gsplat --no-binary gsplat")
for key in _models.keys():
    _models[key] = _models[key].to(device)
print(f"  [启动] 模型已搬到 {device}, +{time.perf_counter() - _t0:.1f}s")

_models['image_super_net'] = imageSuperNet(config=edict(realesrgan_ckpt_path='/root/code/3diclight/trelli_ckpts/ckpts/weights/RealESRGAN_x4plus.pth'))
print(f"  [启动] RealESRGAN 加载完成, +{time.perf_counter() - _t0:.1f}s")

_renderer = render_utils_rl.get_renderer(
    resolution=512,
    near=1,
    far=3,
    bg_color=(1, 1, 1),
    ssaa=1,
)
_hdri_processor = HDRI_Preprocessor(envmap_h=512, envmap_w=1024)
print(f"  [启动] 全部模型/渲染器就绪, 总耗时 {time.perf_counter() - _t0:.1f}s，即将启动 Gradio 服务…")

# rendering options
RENDERING_OPTIONS = {
    "resolution": 512,
    "near": 1,
    "far": 3,
    "bg_color": torch.tensor([0, 0, 0]),
    "ssaa": 1,
    "distributed": False,
}

# 缓存文件夹
CACHE_DIR = "gradio_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Tone Mapper views
AVAILABLE_VIEWS = tone_mapper.available_views

# ========================= 工具函数 =========================
def on_app_load():
    """在应用加载时填充 Tone Mapper 下拉菜单。"""
    if AVAILABLE_VIEWS:
        return gr.update(choices=AVAILABLE_VIEWS, value=AVAILABLE_VIEWS[0])
    return gr.update(choices=[], value=None)

def set_tone_mapper(tone_mapper_select):
    """设置全局 tone mapper 的 view 属性。"""
    if tone_mapper:
        tone_mapper.view = tone_mapper_select
        print(f"Tone Mapper view has been set to: {tone_mapper.view}")

def update_hdri_state(render_state, hdri_file_obj, current_hdri_rot, progress=gr.Progress(track_tqdm=True)):
    """实时更新HDRI，无需重新生成主流程。"""
    if hdri_file_obj is None:
        return None, render_state
    progress(0, desc="读取新的HDRI文件...")
    try:
        new_hdri_np = pyexr.read(hdri_file_obj.name)
        preview_img = tone_mapper.hdr_to_ldr(new_hdri_np)
        # preview_img = np.clip(new_hdri_np ** (1/2.2), 0, 1)
        # preview_img = np.clip(new_hdri_np, 0, 1)
    except Exception as e:
        print(f"无法读取或预览EXR文件: {e}")
        return None, render_state
    if not render_state:
        return preview_img, render_state
    progress(0.5, desc="HDRI已更新。重新编码环境光...")
    render_state['hdri_np'] = new_hdri_np

    hdri_cond = _hdri_processor.preprcess_envir_map(new_hdri_np, current_hdri_rot)
    hdri_cond = torch.cat([hdri_cond[0], hdri_cond[1], hdri_cond[2]], dim=0).float()

    render_state['hdri_cond'] = hdri_cond
    render_state['last_hdri_rot'] = current_hdri_rot
    progress(1.0, desc="环境光更新完毕！")
    return preview_img, render_state

def process_hdri(hdri, hdri_rot=0.):
    hdri = cv2.resize(hdri, (1024, 512), interpolation=cv2.INTER_NEAREST)
    hdri = torch.from_numpy(hdri)
    hdri_rot = [0, 0, hdri_rot]
    envir_map_ldr, envir_map_hdr, envir_map_perceptual, envir_map_hdr_raw, view_dirs_world = _hdri_processor.preprcess_envir_map(
        hdri, hdri_rot)
    hdri_cond = torch.cat([envir_map_ldr, envir_map_hdr, view_dirs_world], dim=0).float()
    return hdri_cond

def encoder_hdri(hdri, hdri_rot=0.):
    hdri_cond = process_hdri(hdri, hdri_rot)
    return _models['hdri_encoder'](hdri_cond[None].to(device))

def get_hdri_roll_video(hdri, num_frames=50):
    hdri_rots = np.linspace(0, 2 * np.pi, num_frames)
    ldr_frames = []
    for hdri_rot in hdri_rots:
        ldr_image = _hdri_processor.rotate_hdri_and_get_cond(hdri, hdri_rot, tone_mapper)  # [3, H, W]
        # 转换为 [H, W, 3] 并归一化到 [0, 255]
        # ldr_image = ldr.permute(1, 2, 0).cpu().numpy()
        # ldr_image = np.clip(ldr_image, 0, 1)
        ldr_image = (ldr_image * 255).astype(np.uint8)
        ldr_frames.append(ldr_image)
    return ldr_frames

def load_slat_data(slat_path: str, device: str) -> sp.SparseTensor:
    """加载SLaT潜在表示数据"""
    loaded_data = np.load(slat_path, allow_pickle=True)

    feats = torch.from_numpy(loaded_data['feats']).to(torch.float32)
    # feats_color = torch.from_numpy(loaded_data['color_feats']).to(torch.float32)
    # feats_brm = torch.from_numpy(loaded_data['brm_feats'][:, :8]).to(torch.float32)
    coords = torch.from_numpy(loaded_data['coords']).to(torch.int32)

    # 添加批次维度到坐标
    coords = torch.cat([
        torch.zeros((coords.shape[0], 1), device=coords.device, dtype=coords.dtype),
        coords
    ], dim=-1)

    # 合并所有特征
    # feats = torch.cat([feats, feats_color, feats_brm], dim=-1)

    return sp.SparseTensor(feats, coords).to(device)


def generate_cameras_spiral(num_views: int, r=2, fov=40) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """生成yaw和pitch的螺旋序列"""
    yaws = torch.linspace(0, 2 * np.pi, num_views)
    pitchs = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * np.pi, num_views))
    yaws = yaws.tolist()
    pitchs = pitchs.tolist()
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    return extrinsics, intrinsics

def generate_camera(yaw: float, pitch: float, r: float, fov: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成单个相机"""
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([yaw], [pitch], r, fov)
    return extrinsics[0], intrinsics[0]

def to_image(img_np):
    """Convert numpy array to PIL Image, handling grayscale."""
    if img_np.ndim == 2:  # Grayscale image
        return Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8), mode='L')
    else:
        return Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8))

@torch.inference_mode()
def process_scene(
    slat_path: str,
    hdri_file_obj: gr.File,
    hdri_rot: float,
    yaw: float,
    pitch: float,
    fov: float,
    radius: float,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]:
    """处理场景的重新照明，并返回预览图片。"""
    # 清理GPU缓存
    torch.cuda.empty_cache()
    # 创建缓存目录
    scene_cache_dir = os.path.join(CACHE_DIR, "scene_cache")
    os.makedirs(scene_cache_dir, exist_ok=True)

    # 加载SLaT数据
    progress(0, desc="加载SLaT数据...")
    slat = load_slat_data(slat_path, device)

    # 加载HDRI数据
    progress(0.1, desc="加载HDRI数据...")
    hdri = pyexr.read(hdri_file_obj.value)[..., :3]
    hdri_cond = encoder_hdri(hdri, hdri_rot * np.pi / 180.)

    # 生成相机
    progress(0.2, desc="生成相机...")
    yaw = yaw * np.pi / 180.
    pitch = pitch * np.pi / 180.
    extr, intr = generate_camera(yaw, pitch, radius, fov)

    # 转换为tensor并移动到GPU
    extr = extr.to(device)
    intr = intr.to(device)

    # 处理HDRI条件
    hdri_cond = hdri_cond.repeat(1, 1, 1)

    # 生成高斯表示
    hs, rfs = _models['decoder'](slat)
    reps = _models['renderer'](hs,rfs,hdri_cond,extr[None,...])

    res = None
    for rep in reps:
        res = _renderer.render(rep, extr, intr, opt=edict(neural_basis=_models['neural_basis']))
    if res is None:
        raise ValueError("Rendering failed: No result returned.")

    pred_image = res['color'].detach().cpu().numpy().transpose(1, 2, 0)
    pred_image = tone_mapper.hdr_to_ldr(pred_image)

    alpha_image = res['alpha_view'].detach().cpu().numpy().transpose(1, 2, 0)

    color_alpha_image = (np.concatenate([pred_image, alpha_image], axis=-1) * 255.0).astype(np.uint8)
    color_alpha_image_sr = _models['image_super_net'](color_alpha_image)

    # resize to 512x512
    color_image = color_alpha_image_sr.resize((512, 512), Image.Resampling.LANCZOS)

    color_image.save(os.path.join(scene_cache_dir, 'color.png'))

    base_color = res['base_color'].detach().cpu().numpy().transpose(1, 2, 0)
    base_color = (np.concatenate([base_color, alpha_image], axis=-1))
    base_color_image = to_image(base_color)
    base_color_image.save(os.path.join(scene_cache_dir, 'base_color.png'))

    metallic = res['metallic'].detach().cpu().numpy().transpose(1, 2, 0)
    metallic = (np.concatenate([metallic.repeat(3, axis=-1), alpha_image], axis=-1))
    metallic_image = to_image(metallic)
    metallic_image.save(os.path.join(scene_cache_dir, 'metallic.png'))

    roughness = res['roughness'].detach().cpu().numpy().transpose(1, 2, 0)
    roughness = (np.concatenate([roughness.repeat(3, axis=-1), alpha_image], axis=-1))
    roughness_image = to_image(roughness)
    roughness_image.save(os.path.join(scene_cache_dir, 'roughness.png'))    

    shadow = res['shadow'].detach().cpu().numpy().transpose(1, 2, 0)
    shadow = (np.concatenate([shadow.repeat(3, axis=-1), alpha_image], axis=-1))
    shadow_image = to_image(shadow)
    shadow_image.save(os.path.join(scene_cache_dir, 'shadow.png'))
    

    return color_image, base_color_image, metallic_image, roughness_image, shadow_image

@torch.inference_mode()
def generate_video(
    slat_path: str,
    hdri_file_obj: gr.File,
    hdri_rot: float,
    fps: int,
    num_views: int,
    fov: float,
    radius: float,
    full_video: bool = False,
    shadow_video: bool = False,
    progress=gr.Progress(track_tqdm=True),
) -> str:
    """生成视频并保存到缓存目录。"""
    # 清理GPU缓存
    torch.cuda.empty_cache()
    # 创建缓存目录
    scene_cache_dir = os.path.join(CACHE_DIR, "scene_cache")
    os.makedirs(scene_cache_dir, exist_ok=True)

    # 加载SLaT数据
    progress(0, desc="加载SLaT数据...")
    slat = load_slat_data(slat_path, device)

    # 加载HDRI数据
    progress(0.1, desc="加载HDRI数据...")
    hdri = pyexr.read(hdri_file_obj.name)[..., :3]
    hdri_cond = encoder_hdri(hdri, hdri_rot * np.pi / 180.)

    # 生成相机
    progress(0.2, desc="生成相机...")
    extrinsics, intrinsics = generate_cameras_spiral(num_views, radius, fov)
    num_views = len(extrinsics)

    # 转换为tensor并移动到GPU
    extrinsics = torch.stack(extrinsics, dim=0).to(device)
    intrinsics = torch.stack(intrinsics, dim=0).to(device)

    # 处理HDRI条件
    hdri_conds = hdri_cond.repeat(num_views, 1, 1)

    results_list = []

    # 对每个视角进行渲染
    progress(0.3, desc="开始渲染...")
    hs, rfs = _models['decoder'](slat)
    start_time = time.time()
    for i, (extr, intr, hdri_cond) in enumerate(zip(extrinsics, intrinsics, hdri_conds)):
        # 生成高斯表示
        reps = _models['renderer'](hs,rfs,hdri_cond[None,...],extr[None,...])

        for rep in reps:
            res = _renderer.render(rep, extr, intr, opt=edict(neural_basis=_models['neural_basis']))
            color = res['color'].detach().cpu().numpy().transpose(1, 2, 0)
            alpha = res['alpha_view'].detach().cpu().numpy().transpose(1, 2, 0)
            color = tone_mapper.hdr_to_ldr(color)
            if not full_video:
                pred_image = (color * alpha + (1 - alpha) * 1)
                results_list.append((pred_image * 255).astype(np.uint8))
            else:
                color = color * alpha + (1 - alpha) * 1
                base_color = res['base_color'].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * 1
                metallic = res['metallic'].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * 1
                roughness = res['roughness'].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * 1
                shadow = res['shadow'].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * 1
                pred_image = np.concatenate([color, base_color, metallic.repeat(3, axis=-1), roughness.repeat(3, axis=-1), shadow.repeat(3, axis=-1)], axis=1) if shadow_video else np.concatenate([color, base_color, metallic.repeat(3, axis=-1), roughness.repeat(3, axis=-1)], axis=1)
                results_list.append((pred_image * 255).astype(np.uint8))

        progress(0.3 + (i / num_views) * 0.7, desc=f"渲染视角 {i + 1}/{num_views}...")
    end_time = time.time()
    print(f"渲染时间: {end_time - start_time} 秒")
    print(f"渲染帧率: {num_views / (end_time - start_time)} 帧/秒")

    # 保存视频
    output_path = os.path.join(scene_cache_dir, 'result.mp4') if not full_video else os.path.join(scene_cache_dir, 'result_full.mp4')
    imageio.mimsave(output_path, results_list, fps=fps)

    return output_path

@torch.inference_mode()
def generate_video_with_hdri(
    slat_path: str,
    hdri_file_obj: gr.File,
    num_frames: int,
    fps: int,
    yaw: float,
    pitch: float,
    fov: float,
    radius: float,
    full_video: bool = False,
    shadow_video: bool = False,
    progress=gr.Progress(track_tqdm=True),
):
    """生成视频并保存到缓存目录, hdri rotated, fixed view。"""
    # 清理GPU缓存
    torch.cuda.empty_cache()
    # 创建缓存目录
    scene_cache_dir = os.path.join(CACHE_DIR, "scene_cache")
    os.makedirs(scene_cache_dir, exist_ok=True)

    # 加载SLaT数据
    progress(0, desc="加载SLaT数据...")
    slat = load_slat_data(slat_path, device)

    # 加载HDRI数据
    progress(0.1, desc="加载HDRI数据...")
    hdri = pyexr.read(hdri_file_obj.name)[..., :3]
    hdri_rots = 2 * np.pi * np.arange(num_frames) / num_frames

    hdri_conds = []
    for hdri_rot in hdri_rots:
        hdri_cond = encoder_hdri(hdri, hdri_rot)
        hdri_conds.append(hdri_cond)
    hdri_conds = torch.stack(hdri_conds, dim=0)

    # Gen hdri roll video
    progress(0.15, desc="生成HDRI roll视频...")
    hdri_roll_video = get_hdri_roll_video(hdri, num_frames)
    hdri_roll_video_path = os.path.join(scene_cache_dir, 'hdri_roll_video.mp4')
    imageio.mimsave(hdri_roll_video_path, hdri_roll_video, fps=fps)

    # 生成相机
    progress(0.2, desc="生成相机...")
    yaw = yaw * np.pi / 180.
    pitch = pitch * np.pi / 180.
    extrinsic, intrinsic = generate_camera(yaw, pitch, radius, fov)

    extrinsic = extrinsic.to(device)
    intrinsic = intrinsic.to(device)

    results_list = []

    progress(0.3, desc="开始渲染...")
    hs, rfs = _models['decoder'](slat)
    for i, hdri_cond in enumerate(hdri_conds):
        reps = _models['renderer'](hs,rfs,hdri_cond,extrinsic[None,...])
        for rep in reps:
            res = _renderer.render(rep, extrinsic, intrinsic, opt=edict(neural_basis=_models['neural_basis']))
            color = res['color'].detach().cpu().numpy().transpose(1, 2, 0)
            alpha = res['alpha_view'].detach().cpu().numpy().transpose(1, 2, 0)
            # white bg
            color = tone_mapper.hdr_to_ldr(color) 
            if not full_video:
                pred_image = (color * alpha + (1 - alpha) * 1)
                results_list.append((pred_image * 255).astype(np.uint8))
            else:
                color = color * alpha + (1 - alpha) * 1
                base_color = res['base_color'].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * 1
                metallic = res['metallic'].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * 1
                roughness = res['roughness'].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * 1
                shadow = res['shadow'].detach().cpu().numpy().transpose(1, 2, 0) * alpha + (1 - alpha) * 1
                pred_image = np.concatenate([color, base_color, metallic.repeat(3, axis=-1), roughness.repeat(3, axis=-1), shadow.repeat(3, axis=-1)], axis=1) if shadow_video else np.concatenate([color, base_color, metallic.repeat(3, axis=-1), roughness.repeat(3, axis=-1)], axis=1)
                results_list.append((pred_image * 255).astype(np.uint8))

    progress(0.9, desc="合成视频...")
    output_path = os.path.join(scene_cache_dir, 'result_hdri_rot.mp4') if not full_video else os.path.join(scene_cache_dir, 'result_hdri_rot_full.mp4')
    imageio.mimsave(output_path, results_list, fps=fps)

    return hdri_roll_video_path, output_path

def generate_3d(
    # input_image_np: np.ndarray,
    slat_path: str, 
    # mesh_path: str,
    hdri_file_obj: gr.File, 
    hdri_rot: float,
    simplify: float,
    tex_size: int,
    # use_realesrgan: bool, 
    progress=gr.Progress(track_tqdm=True)
):
    scene_cache_dir = os.path.join(CACHE_DIR, "scene_cache")
    os.makedirs(scene_cache_dir, exist_ok=True)

    progress(0.05, desc="加载基础网格中...")
    # image = Image.fromarray(input_image_np).convert("RGBA")
    # if image.getbands() == ('R', 'G', 'B'): 
        # image = rembg(image)
    # base_mesh = pipeline_shapegen(image=image)[0]

    # base_mesh_path = os.path.join(scene_cache_dir, 'base_mesh.glb')
    # base_mesh.export(base_mesh_path)
    # base_mesh = trimesh.load(mesh_path)

    progress(0.15, desc="读取HDRI中...")
    hdri = pyexr.read(hdri_file_obj.name)[..., :3]
    with torch.inference_mode():
        progress(0.25, desc="编码HDRI中...")
        hc = encoder_hdri(hdri, hdri_rot)

        progress(0.30, desc="载入结构化隐变量中...")
        slat = load_slat_data(slat_path, device = device)
        # slat_mesh = load_slat_data(slat_path, feats_type = 'e', device = device)

        progress(0.50, desc="解码PBR特征中...")
        h, rf = _models['decoder'](slat)

        mesh_ori = _models['slat_decoder_mesh'](slat)
        base_mesh = mesh_ori[0]

    # 导出GLB文件
    progress(0.75, desc="生成GLB网格和材质...")
    mesh_export = render_utils_rl.to_glb(
        _models['renderer'], 
        h, rf, hc, 
        _models['neural_basis'],
        base_mesh, 
        tone_mapper=tone_mapper, 
        simplify=simplify, 
        fill_holes=True, 
        texture_size=int(tex_size)
    )
    
    glb_mesh_path = os.path.join(scene_cache_dir, 'pbr.glb')
    
    progress(0.95, desc="导出GLB文件...")
    mesh_export.export(glb_mesh_path)

    progress(1.0, desc="完成！")

    return glb_mesh_path, gr.update(value=glb_mesh_path, visible=True)

def clear_cache():
    """清除缓存文件夹。"""
    try:
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)
        return "缓存已清除！"
    except Exception as e:
        return f"清除缓存失败: {e}"

# ========================= Gradio界面 =========================
with gr.Blocks(title="3D Relighting with TRELLIS") as demo:
    gr.Markdown("# 3D Relighting with TRELLIS")

    # Tone Mapper 选择
    with gr.Row():
        tone_mapper_select = gr.Dropdown(choices=AVAILABLE_VIEWS, value=AVAILABLE_VIEWS[0], label="Tone Mapper")

    # HDRI 相关
    with gr.Row():
        hdri_file = gr.File(label="上传 HDRI (EXR)", file_types=[".exr"])
        hdri_preview = gr.Image(label="HDRI 预览", height=256)
    hdri_rotation = gr.Slider(minimum=0, maximum=360, value=0, step=1, label="HDRI 旋转角度")
    hdri_state = gr.State()

    # SLaT Path
    slat_path = gr.Textbox(label="SLaT Path", value="/baai-cwm-vepfs/cwm/hong.li/code/3dgen/data/zeroverse/render_all/gs_benchmark/pbr2dinov2_voxel_64_slat/bag.npz")

    with gr.Row():
        # 相机参数
        with gr.Accordion("相机参数", open=True):
            yaw = gr.Slider(minimum=0., maximum=360., value=0, step=0.1, label="Yaw")
            pitch = gr.Slider(minimum=-90., maximum=90., value=0, step=0.1, label="Pitch")
            fov = gr.Slider(minimum=10, maximum=70, value=40, step=1, label="Field of View (FOV)")
            radius = gr.Slider(minimum=1, maximum=10, value=2, step=0.1, label="Radius")

        # 视频参数
        with gr.Accordion("视频参数", open=True):
            num_views = gr.Slider(minimum=3, maximum=120, value=100, step=1, label="Number of Views for Camera Path (video)")
            num_frames = gr.Slider(minimum=3, maximum=120, value=100, step=1, label="Number of Rots for HDRI (video)")
            fps = gr.Slider(minimum=1, maximum=100, value=25, step=1, label="FPS")

    # 预览和生成
    with gr.Row():
        preview_button = gr.Button("生成预览图片")
        generate_button = gr.Button("生成视频")
        generate_button_hdri_rot = gr.Button("生成视频(HDRI旋转)")

    # 输出
    with gr.Row():
        color_output = gr.Image(label="Color", show_label=True)
        base_color_output = gr.Image(label="Base Color", show_label=True)
        metallic_output = gr.Image(label="Metallic", show_label=True)
        roughness_output = gr.Image(label="Roughness", show_label=True)
        shadow_output = gr.Image(label="Shadow", show_label=True)
    with gr.Row():
        video_output = gr.Video(label="camera path variation video", autoplay=True)
        hdri_roll_video_output = gr.Video(label="HDRI roll video", autoplay=True)
        video_output_hdri_rot = gr.Video(label="video with hdri rotated", autoplay=True)
    
    with gr.Row():
        generate_button_full = gr.Button("gen_full_video_path_variation_video")
        generate_button_hdri_rot_full = gr.Button("gen_full_video_hdri_rotated_video")

    full_video = gr.Checkbox(label="full video", value=True)
    shadow_video = gr.Checkbox(label="shadow video", value=False)

    with gr.Row():
        full_video_output_path = gr.Video(label="full video path variation video", autoplay=True)
    with gr.Row():
        full_video_hdri_rot_path = gr.Video(label="full video with hdri rotated", autoplay=True)

    with gr.Row():
        # mesh_path = gr.Textbox(label="Mesh Path", value="/baai-cwm-vepfs/public_data/rendering_data/3diclight/3diclight_rendering/3diclight_even_8w9/meshs/715921f8706f4c9219575645a9a32b327efefc34de128ecf96f5f3f96cb48871/mesh.ply")
        # input_image_np = gr.Image(label="输入图像", height=512, width=512)
        with gr.Column(scale=1):
            simplify_level = gr.Slider(0.0, 1.0, 0.95, step=0.05, label="网格简化")
            texture_size = gr.Slider(512, 4096, 2048, step=512, label="贴图分辨率")
            export_glb_btn = gr.Button("导出GLB", variant="secondary")
            output_glb_file = gr.File(label="下载导出的GLB文件", interactive=False)
        with gr.Column(scale=2):
            # output_glb_preview = gr.Model3D(label="最终导出的GLB预览", interactive=False, height=300)
            output_glb_preview = gr.Model3D(label="最终导出的GLB预览", interactive=False, height=300)
            # hdri_file.change(
                # lambda x: gr.update(env_map=x.name if x is not None else None),
                # inputs=hdri_file, outputs=[output_glb_preview])

    # 清除缓存
    with gr.Row():
        clear_cache_button = gr.Button("清除缓存")
        cache_status = gr.Textbox(label="缓存状态")

    # 函数绑定
    demo.load(on_app_load, outputs=[tone_mapper_select])
    tone_mapper_select.change(set_tone_mapper, inputs=[tone_mapper_select])
    hdri_rotation.release(update_hdri_state, inputs=[hdri_state, hdri_file, hdri_rotation], outputs=[hdri_preview, hdri_state])
    hdri_file.upload(update_hdri_state, inputs=[hdri_state, hdri_file, hdri_rotation], outputs=[hdri_preview, hdri_state])

    preview_button.click(
        fn=process_scene,
        inputs=[slat_path, hdri_file, hdri_rotation, yaw, pitch, fov, radius],
        outputs=[color_output, base_color_output, metallic_output, roughness_output, shadow_output],
    )

    generate_button.click(
        fn=generate_video,
        inputs=[slat_path, hdri_file, hdri_rotation, fps, num_views, fov, radius],
        outputs=[video_output],
    )

    generate_button_hdri_rot.click(
        fn=generate_video_with_hdri,
        inputs=[slat_path, hdri_file, num_frames, fps, yaw, pitch, fov, radius],
        outputs=[hdri_roll_video_output, video_output_hdri_rot],
    )

    generate_button_full.click(
        fn=generate_video,
        inputs=[slat_path, hdri_file, hdri_rotation, fps, num_views, fov, radius, full_video, shadow_video],
        outputs=[full_video_output_path],
    )

    generate_button_hdri_rot_full.click(
        fn=generate_video_with_hdri,
        inputs=[slat_path, hdri_file, num_frames, fps, yaw, pitch, fov, radius, full_video, shadow_video],
        outputs=[hdri_roll_video_output, full_video_hdri_rot_path],
    )

    export_glb_btn.click(
        generate_3d, 
        # [input_image_np, slat_path, hdri_file, hdri_rotation, simplify_level, texture_size], 
        [slat_path, hdri_file, hdri_rotation, simplify_level, texture_size], 
        [output_glb_file, output_glb_preview]
    )

    clear_cache_button.click(
        fn=clear_cache,
        inputs=[],
        outputs=[cache_status],
    )

# ========================= 启动 Gradio =========================
# 首次打开网页很慢的原因：上面在 import 阶段就加载了 5 个 TRELLIS 模型 + RealESRGAN，
# 必须等它们全部加载完才会执行到 launch() 并绑定端口，浏览器才能连上。若要“先出页面再加载模型”需改为懒加载。
if __name__ == "__main__":
    demo.queue(max_size=5).launch(
        server_name="0.0.0.0",
        server_port=7294,
        share=True,  # required when localhost is not accessible (e.g. remote server)
    )