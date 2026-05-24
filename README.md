# NeAR

<div align="center">
  <img src="https://near-project.github.io/static/logo.svg" alt="NeAR Logo" width="320"/>
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2511.18600"><img src="https://img.shields.io/badge/2511.18600-arXiv-b31b1b?logo=arxiv&logoColor=red" alt="arXiv"></a>
  <a href="https://near-project.github.io/"><img src="https://img.shields.io/badge/Project_Page-Website-2ea44f?logo=googlechrome&logoColor" alt="Project Page"></a>
  <a href="https://huggingface.co/luh0502/NeAR/tree/main"><img src="https://img.shields.io/badge/HuggingFace-Weights-f9d949?logo=huggingface&logoColor" alt="Checkpoints"></a>
  <a href="https://huggingface.co/spaces/luh0502/NeAR"><img src="https://img.shields.io/badge/HuggingFace-Demo-f9d949?logo=huggingface&logoColor" alt="Demo"></a>
  <a href="https://www.modelscope.cn/datasets/luh0502/NeAR-dataset"><img src="https://img.shields.io/badge/ModelScope-Dataset-624aff?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTI4IiBoZWlnaHQ9IjEyOCIgdmlld0JveD0iMCAwIDEyOCAxMjgiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTY0IDEyTDExMCAzOC42djUyLjhMNjQgMTE4IDE4IDkxLjRWMzguNkw2NCAxMloiIGZpbGw9IiNmZmYiLz48cGF0aCBkPSJNNjQgMzAuNkwzNC4xIDQ3Ljl2MzQuMkw2NCA5OS40bDI5LjktMTcuM1Y0Ny45TDY0IDMwLjZaIiBmaWxsPSIjNjI0YWZmIi8+PHBhdGggZD0iTTY0IDQ2LjJMNDcuNiA1NS43djE4LjZMNjQgODMuOGwxNi40LTkuNVY1NS43TDY0IDQ2LjJaIiBmaWxsPSIjZmZmIi8+PC9zdmc+" alt="ModelScope Dataset"></a>
</div>

**NeAR** is a relightable 3D generation and rendering project built on top of **TRELLIS-style Structured Latents (SLAT)** and a lighting-aware neural renderer. Given a casually lit input image, NeAR estimates relightable neural assets and renders them under novel environment lighting and viewpoints.

This repository combines:

- a **TRELLIS-derived latent pipeline** for image-conditioned SLAT prediction,
- a **lighting-aware neural renderer** conditioned on HDR environment maps,
- an optional **geometry frontend** based on **Hunyuan3D-2.1**,
- tools for **single-view relighting**, **novel-view relighting**, **HDRI rotation videos**, and **GLB export**.

<!-- For more details, please check the [project page](https://near-project.github.io/), the [paper on arXiv](https://arxiv.org/abs/2511.18600), and the [Hugging Face model repository](https://huggingface.co/luh0502/NeAR/tree/main). -->

## Release Status

- [✓] Checkpoints / model weights
- [✓] Inference code
- [✓] Hugging Face demo
- [✓] Data release
- [ ] Training code

## News
- Inference code and Checkpoints have been released! 
- ⭐ **2025.04** — NeAR has been selected as a **Highlight** at **CVPR 2026**! 
- The Hugging Face demo is currently being deployed.
- Data and training code are coming soon.

## Teaser

<div align="center">
  <img src="assets/teaser/teaser.gif" alt="NeAR teaser" width="100%" />
</div>

**Relightable 3D generative rendering results.** Columns from left to right depict the target illumination, the casually lit input image, Blender-rendered results from Trellis 3D, Hunyuan 3D-2.1 (with PBR materials), our method's estimated multi-view PBR materials back-projected onto the given mesh, our neural rendering results, and ground truth.

## Example Relighting / Material Videos

The following videos are produced by the local NeAR example pipeline and are useful for quickly previewing:

- **Novel-view relighting video**: camera moves while the illumination stays fixed.
- **HDRI rotation preview**: environment map rotates while the camera stays fixed.
- **Relighting under rotating HDRI**: material response changes under time-varying illumination.

<div align="center">
  <img src="assets/pbr_vis/pbrvideo.gif" alt="NeAR material and relighting visualization" width="100%" />
</div>

<!-- If these local videos are not present, you can generate them with `example.py` and `--video_frames > 0`. -->

## Texture Style Transfer

Different reference images can be used for **mesh** and **SLaT** in the Gradio app — the mesh is locked to one image's geometry while the SLaT inherits the appearance of another, producing texture style transfer onto the same shape.

<div align="center">
  <img src="assets/style/style.gif" alt="NeAR texture style transfer" width="100%" />
</div>

---

## Overview

NeAR couples **asset representation** and **renderer design**:

- **Asset side**: from an input image, a structured latent representation stores geometry-aware and material-aware information in a compact sparse latent.
- **Renderer side**: a neural renderer takes the latent, view parameters, and an HDR environment map, then predicts relightable outputs such as color, base color, metallic, roughness, and shadow.

Compared with a standard image-to-3D pipeline, NeAR focuses on:

- **relighting under novel HDR illumination**,
- **view-consistent rendering**,
- **fast feed-forward inference**, and
- **material-aware rendering outputs**.

---

## Repository Structure

Key files and directories:

- `example.py` — minimal end-to-end inference example.
- `app_e.py` — Gradio-style demo / app script.
- `app_viser.py` — interactive neural relight viewer ([viser](https://github.com/viser-project/viser)); orbit camera + HDRI controls, full-viewport relit RGB (no GLB).
- `pixi.toml` — reproducible environment definition (conda + PyPI, CUDA toolchain).
- `checkpoints/` — local pipeline configuration and model checkpoints.
- `trellis/pipelines/near_image_to_relightable_3d.py` — main NeAR inference pipeline.
- `trellis/utils/render_utils_rl.py` — relighting rendering utilities.
- `trellis/datasets/hdri_processer.py` — HDRI preprocessing and rotation helpers.
- `hy3dshape/` — Hunyuan3D shape utilities from [Tencent-Hunyuan/Hunyuan3D-2.1/hy3dshape](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/tree/main/hy3dshape).

---

## Installation

### Requirements

- Linux (tested on Ubuntu with glibc ≥ 2.35)
- NVIDIA GPU with CUDA 12.x compatible driver (Ampere / Hopper recommended; arch list `8.0;8.6;9.0`)
- [pixi](https://pixi.sh/) ≥ 0.40 — `curl -fsSL https://pixi.sh/install.sh | bash`

The environment uses **Python 3.10**, **PyTorch 2.11.0+cu128**, and **xformers 0.0.35**. All Python / conda dependencies (CUDA toolkit, compilers, system libs) are declared in `pixi.toml`; you do not need a system-wide CUDA install or `conda`.

### Setup

```bash
git clone --recursive https://github.com/Luh1124/NeAR.git
cd NeAR

# 1. Resolve & install python, CUDA toolkit, torch, xformers, and all pure-python deps.
pixi install

# 2. (Optional) Stage Hunyuan3D-2.1's hy3dshape/ at the repo root for geometry generation.
pixi run fetch-hy3dshape

# 3. Build CUDA / torch-extension deps that need torch present at compile time:
#      flash-attn, gsplat, vox2seq (local), diffoctreerast, hy3dshape requirements.
pixi run setup-cuda-ext
```

That's the whole install. Everything below assumes you prefix commands with `pixi run` (or use `pixi shell` for an interactive session):

```bash
pixi run python example.py --image assets/example_image/T.png \
                           --hdri  assets/hdris/studio_small_03_1k.exr
```

### Notes on the environment

- **Cache placement** — `pixi.toml` sets `UV_CACHE_DIR=$PIXI_PROJECT_ROOT/.pixi/uv-cache` so the PyPI cache lives on the same filesystem as `.pixi/envs/`. This lets uv hardlink wheels into the env and avoids cross-mount copy fallbacks.
- **Switching torch versions** — every CUDA extension (`nvdiffrast`, `flash-attn`, `vox2seq`, `diffoctreerast`, `gsplat`) is compiled against a specific torch ABI. CUDA extensions are installed via `pixi run setup-cuda-ext` with `--no-cache-dir --force-reinstall`, so re-running the task after a torch pin change rebuilds them against the new ABI.
- **kaolin** — NVIDIA only publishes wheels up to torch 2.8.0_cu128, so kaolin is *not* installed in this env. No main-path NeAR code imports kaolin (only unused `flexicubes/examples/` demos do). If you need it, build from source.
- **HIP / ROCm** — not currently supported by the pixi env.

### Pipeline configuration

The local pipeline configuration is defined in:

- `checkpoints/pipeline.yaml`

It references the main model components used by NeAR, including:

- `decoder`
- `hdri_encoder`
- `neural_basis`
- `renderer`
- `slat_flow_model`

The geometry model is currently run separately in `example.py` via:

- `tencent/Hunyuan3D-2.1`

### Data

- [luh0502/NeAR](https://huggingface.co/datasets/luh0502/NeAR) — stage-1 dataset on HuggingFace
- [luh0502/NeAR-dataset](https://www.modelscope.cn/datasets/luh0502/NeAR-dataset) — stage-2 dataset on ModelScope

### HDR Environment Maps

Preprocessed HDR environment maps used for training and inference:

- [luh0502/hdr_envmaps_exr_1K](https://huggingface.co/datasets/luh0502/hdr_envmaps_exr_1K) — 1K resolution, normalized to 0–65536 float EXR

---

## Inference

NeAR supports two inference paths:

1. **Image → relightable result** — preprocess image → generate geometry (Hunyuan3D) → predict SLAT → render under target HDRI.
2. **Existing SLaT → relightable result** — skip geometry/latent generation, render directly from a saved `.npz`.

For detailed instructions, command-line examples, output descriptions, and API usage, see [**doc/infer.md**](doc/infer.md).

Quick start:

```bash
pixi run python example.py \
  --image assets/example_image/T.png \
  --hdri assets/hdris/studio_small_03_1k.exr \
  --out_dir relight_out
```

## Acknowledgements

This repository builds on and adapts ideas, codebases, and problem settings from several recent works on structured 3D latents, relighting, inverse rendering, and PBR-aware 3D generation, including:

- [TRELLIS](https://github.com/microsoft/TRELLIS) — structured latent generation and sparse 3D asset representations
- [Hunyuan3D 2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) — image-to-geometry generation and image examples
- [DiLightNet](https://dilightnet.github.io/) — diffusion-based lighting control
- [Neural Gaffer](https://neural-gaffer.github.io/) — object relighting
- [DiffusionRenderer](https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/) — neural inverse / forward rendering
- [MeshGen](https://heheyas.github.io/MeshGen/) — PBR textured mesh generation
- [RGB↔X](https://zheng95z.github.io/publications/rgbx24) — material- and lighting-aware decomposition and synthesis

## BibTeX

If you find this project or data useful, please consider citing our paper:

```bibtex
@inproceedings{li2025near,
  title={NeAR: Coupled Neural Asset-Renderer Stack},
  author={Li, Hong and Ye, Chongjie and Chen, Houyuan and Xiao, Weiqing and Yan, Ziyang and Xiao, Lixing and Chen, Zhaoxi and Xiang, Jianfeng and Xu, Shaocong and Liu, Xuhui and Wang, Yikai and Zhang, Baochang and Han, Xiaoguang and Yang, Jiaolong and Zhao, Hao},
  booktitle={CVPR},
  year={2026}
}
```
