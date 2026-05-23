# NeAR Training

This directory contains training code for the NeAR asset-renderer stack.

## Overview

Training is divided into two stages:

1. **VAE Training** — Train the Structured Latent VAE (encoder + decoders for Gaussian / mesh / radiance-field).
2. **Diffusion Training** — Train the image-conditioned flow-matching diffusion model that generates SLATs.

Both stages share the same entry point (`train.py`), differing only in config files and datasets.

## Quick Start

### Diffusion (Image → SLAT)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python training/train.py \
  --config training/configs/generation/slat_flow_img_dit_L_64l8p2_fp16_i2e.yaml \
  --output_dir outputs/dev.slat_flow_i2e
```

### VAE

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python training/train.py \
  --config training/configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json \
  --output_dir outputs/dev.slat_vae_gs \
  --data_dir ./data/
```

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--config` | Path to config file. Supports both **JSON** and **YAML** (auto-detected by extension). |
| `--output_dir` | Where checkpoints, logs and samples are saved. |
| `--data_dir` | Root data directory (used by TRELLIS-style datasets; I2E configs specify paths in `dataset.args`). |
| `--pretrained_path` | Optional path to a pretrained base flow model (e.g. `slat_flow_img_dit_L_64l8p2_fp16.safetensors`) for initialization. |
| `--ckpt` | Resume from checkpoint: `latest`, `none`, or a specific step number. |

## Directory Layout

```
training/
├── train.py                  # Unified training entry (VAE & diffusion)
├── run_train.sh              # Example launch commands
├── configs/
│   ├── generation/           # Diffusion configs (YAML/JSON)
│   └── vae/                  # VAE configs (JSON)
├── dataset_toolkits/         # Data pre-processing scripts
│   ├── encode_latent.py
│   ├── extract_feature.py
│   ├── render.py
│   └── ...
└── data_csvs/                # CSV preparation helpers
```

## Data Preparation

1. Prepare your dataset following `dataset_toolkits/`.
2. Generate metadata CSVs using `data_csvs/prepare_data.py`.
3. Update `meta_csv` or `roots` paths in the config file.

## Notes

- `train.py` accepts both **JSON** and **YAML** configs (auto-detected by file extension).
- `--pretrained_path` only affects the `denoiser` model. Set it in config or CLI; empty CLI value will not override a config value.
- Multi-GPU and multi-node training are supported out of the box.
