#!/bin/bash

# Image-conditioned SLAT diffusion (I2E) with tar-based dataset
CUDA_VISIBLE_DEVICES=0,1,2,3 python training/train.py \
  --config training/configs/generation/slat_flow_img_dit_L_64l8p2_fp16_i2e_tar.yaml \
  --output_dir outputs/dev.slat_flow_i2e \
  --pretrained_path /root/code/3diclight/trelli_ckpts/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors

# CSV-based dataset (legacy, requires meta_csv)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python training/train.py \
#   --config training/configs/generation/slat_flow_img_dit_L_64l8p2_fp16_i2e.yaml \
#   --output_dir outputs/dev.slat_flow_i2e_csv

# VAE training example
# CUDA_VISIBLE_DEVICES=0,1,2,3 python training/train.py \
#   --config training/configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json \
#   --output_dir outputs/dev.slat_vae_gs \
#   --data_dir ./data/
