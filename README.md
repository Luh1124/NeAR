# NeAR

<div align="center">
  <img src="https://near-project.github.io/static/logo.svg" alt="NeAR Logo" width="320"/>
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2511.18600"><img src="https://img.shields.io/badge/arXiv-2511.18600-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://near-project.github.io/"><img src="https://img.shields.io/badge/Project_Page-Website-2ea44f?logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://huggingface.co/luh0502/NeAR/tree/main"><img src="https://img.shields.io/badge/HuggingFace-NeAR-f9d949?logo=huggingface&logoColor=black" alt="Hugging Face"></a>
</div>

**NeAR** is a relightable 3D generation and rendering project built on top of **TRELLIS-style Structured Latents (SLAT)** and a lighting-aware neural renderer. Given a casually lit input image, NeAR estimates relightable neural assets and renders them under novel environment lighting and viewpoints.

This repository combines:

- a **TRELLIS-derived latent pipeline** for image-conditioned SLAT prediction,
- a **lighting-aware neural renderer** conditioned on HDR environment maps,
- an optional **geometry frontend** based on **Hunyuan3D-2.1**,
- tools for **single-view relighting**, **novel-view relighting**, **HDRI rotation videos**, and **GLB export**.

<!-- For more details, please check the [project page](https://near-project.github.io/), the [paper on arXiv](https://arxiv.org/abs/2511.18600), and the [Hugging Face model repository](https://huggingface.co/luh0502/NeAR/tree/main). -->

## Release Status

- [x] Checkpoints / model weights
- [x] Inference code
- [ ] Hugging Face demo
- [ ] Data release
- [ ] Training code

> The Hugging Face demo is currently being deployed.
> Data and training code are coming soon.

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

If these local videos are not present, you can generate them with `example.py` and `--video_frames > 0`.

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
- `setup.sh` — environment setup helper.
- `checkpoints/` — local pipeline configuration and model checkpoints.
- `trellis/pipelines/near_image_to_relightable_3d.py` — main NeAR inference pipeline.
- `trellis/utils/render_utils_rl.py` — relighting rendering utilities.
- `trellis/datasets/hdri_processer.py` — HDRI preprocessing and rotation helpers.
- `hy3dshape/` — local Hunyuan3D code used for geometry generation.

---

## Installation

### Requirements

- Linux
- NVIDIA GPU
- Python 3.10+ recommended
- CUDA-compatible PyTorch environment

NeAR inherits many dependencies from TRELLIS and additionally uses relighting-related packages such as `pyexr`, `simple_ocio`, `open3d`, and the local `hy3dshape` module.

### Setup

Use the provided setup script as a starting point:

```bash
cd /root/code/3diclight/NeAR
. ./setup.sh --help
```

A typical TRELLIS-style setup may look like:

```bash
. ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```

Depending on your environment, you may still need to manually install extra packages used by NeAR, for example:

```bash
pip install pyexr simple-ocio open3d rembg imageio easydict
```

If you use Hunyuan3D geometry generation, make sure the `hy3dshape` dependencies are also installed.

---

## Checkpoints

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

---

## Inference Modes

NeAR currently supports two practical inference modes.

### 1. From image to relightable result

Pipeline:

1. preprocess the image,
2. generate geometry using Hunyuan3D,
3. convert geometry to sparse coordinates,
4. predict SLAT from the image and geometry,
5. render under a target HDRI.

### 2. From existing SLaT to relightable result

If you already have a saved `.npz` SLaT file, NeAR can skip geometry and latent generation, and directly render under a target HDRI.

---

## Minimal Example

The main entry point is `example.py`.

### Single-image relighting

```bash
python example.py \
  --image assets/example_image/T.png \
  --hdri assets/hdris/studio_small_03_1k.exr \
  --out_dir relight_out
```

### Rotate the environment light

```bash
python example.py \
  --image assets/example_image/T.png \
  --hdri assets/hdris/studio_small_03_1k.exr \
  --hdri_rot 90 \
  --out_dir relight_out
```

### Render from an existing SLaT

```bash
python example.py \
  --slat /path/to/sample_slat.npz \
  --hdri assets/hdris/studio_small_03_1k.exr \
  --out_dir relight_out
```

### Generate camera-path and HDRI-rotation videos

```bash
python example.py \
  --image assets/example_image/T.png \
  --hdri assets/hdris/studio_small_03_1k.exr \
  --video_frames 40 \
  --out_dir relight_out
```

---

## Example Outputs

Running `example.py` typically produces:

- `relight_out/initial_3d_shape.glb` — geometry generated by Hunyuan3D
- `relight_out/relight_color.png` — relit color result
- `relight_out/base_color.png` — estimated base color
- `relight_out/metallic.png` — metallic map visualization
- `relight_out/roughness.png` — roughness map visualization
- `relight_out/shadow.png` — shadow map visualization
- `relight_out/relight_camera_path.mp4` — novel-view relighting video
- `relight_out/hdri_roll.mp4` — rotating HDRI preview
- `relight_out/relight_hdri_rotation.mp4` — fixed-view relighting under rotating HDRI

If `--save_slat` is specified, the inferred SLaT will also be saved as an `.npz` file.

---

## Important Notes

### 1. Geometry is run outside the main NeAR pipeline

To avoid coupling geometry inference too tightly with the relighting pipeline, the current codebase runs **Hunyuan3D separately in `example.py`**, then passes the generated mesh into:

- `pipeline.run_with_shape(...)`

This design keeps the relighting pipeline cleaner and makes geometry easier to swap out.

### 2. HDRI rotation

`--hdri_rot` in `example.py` controls the **static rotation angle** for regular rendering.

For continuous environment rotation, `example.py` also calls:

- `pipeline.render_hdri_rotation_video(...)`

which returns both:

- rotated HDRI preview frames, and
- rendered relighting frames.

### 3. Full video rendering

Some video modes concatenate multiple outputs side by side:

- color
- base color
- metallic
- roughness
- shadow

This is useful for debugging and qualitative comparison, but increases video width and storage size.

### 4. Resolution and rendering options

The renderer is configured inside the pipeline via:

- `setup_renderer(...)`
- `render_view(...)`
- `render_camera_path_video(...)`
- `render_hdri_rotation_video(...)`

You can adjust:

- output resolution,
- camera FOV,
- camera radius,
- background color,
- HDRI rotation,
- video frame count.

---

## Core API

Main NeAR pipeline methods include:

- `preprocess_image(image)`
- `run_with_shape(image, mesh, ...)`
- `run_with_coords(image_list, coords, ...)`
- `load_slat(path)`
- `load_hdri(path)`
- `render_view(...)`
- `render_camera_path_video(...)`
- `render_hdri_rotation_video(...)`
- `export_glb_from_slat(...)`

---

## Typical Workflow

A practical workflow is:

1. start from an image,
2. generate geometry with Hunyuan3D,
3. infer SLaT with `run_with_shape`,
4. save the SLaT,
5. reuse the SLaT for different HDRIs, different HDRI rotations, and different camera paths.

This avoids recomputing geometry and latent generation every time you want to test a new lighting setup.

---

## Related Projects

- [TRELLIS](https://github.com/microsoft/TRELLIS)
- [NeAR Project Page](https://near-project.github.io)
- [Hunyuan3D](https://huggingface.co/tencent/Hunyuan3D-2.1)
- [DiLightNet](https://dilightnet.github.io/)
- [Neural Gaffer](https://neural-gaffer.github.io/)
- [DiffusionRenderer](https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/)
- [MeshGen](https://heheyas.github.io/MeshGen/)
- [RGB↔X](https://zheng95z.github.io/publications/rgbx24)

---

## Acknowledgements

This repository builds on and adapts ideas, codebases, and problem settings from several recent works on structured 3D latents, relighting, inverse rendering, and PBR-aware 3D generation, including:

- **TRELLIS** for structured latent generation and sparse 3D asset representations,
- **Hunyuan3D 2.1** for image-to-geometry generation,
- **DiLightNet** and **Neural Gaffer** for diffusion-based lighting control and object relighting,
- **DiffusionRenderer** for neural inverse / forward rendering under complex appearance and illumination,
- **MeshGen** for PBR textured mesh generation,
- **RGB↔X** for material- and lighting-aware decomposition and synthesis,
- environment-map-based relighting workflows for HDR-conditioned neural rendering.

We thank the authors of these projects for releasing their papers, code, models, and project pages. If you use this repository, please also check the licenses and terms of the upstream dependencies and models.

## BibTeX

If you find this project useful, please consider citing our paper:

```bibtex
@inproceedings{li2025near,
  title={NeAR: Coupled Neural Asset-Renderer Stack},
  author={Li, Hong and Ye, Chongjie and Chen, Houyuan and Xiao, Weiqing and Yan, Ziyang and Xiao, Lixing and Chen, Zhaoxi and Xiang, Jianfeng and Xu, Shaocong and Liu, Xuhui and Wang, Yikai and Zhang, Baochang and Han, Xiaoguang and Yang, Jiaolong and Zhao, Hao},
  booktitle={CVPR},
  year={2026}
}
```