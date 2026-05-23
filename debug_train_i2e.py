#!/usr/bin/env python3
"""
调试 / 验证 I2E 训练全流程的独立脚本。
加载预训练模型 + SLatI2ETar 数据集，执行一次完整的前向+反向传播。

用法:
    cd /root/code/3diclight/NeAR
    python debug_train_i2e.py
"""

import os
import sys
import time

sys.path.insert(0, '/root/code/3diclight/NeAR')
sys.path.insert(0, '/root/code/3diclight/NeAR/hy3dshape')

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from safetensors.torch import load_file


def main():
    # ======================== 配置 ========================
    pretrained_path = '/root/code/3diclight/trelli_ckpts/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors'
    image_tar_dir = '/root/data/3diclight/downloaded_ref_images/ref_images'
    slat_tar_path = '/root/data/3diclight/LH-Slat.tar'
    batch_size = 2
    image_size = 518
    device = 'cuda'

    print("=" * 60)
    print("[1/6] 初始化数据集")
    print("=" * 60)
    from trellis.datasets import SLatI2ETar

    t0 = time.time()
    dataset = SLatI2ETar(
        image_tar_dir=image_tar_dir,
        slat_tar_path=slat_tar_path,
        cache_index=True,
        max_num_voxels=32768,
        image_size=image_size,
        normalization={
            'mean': [-2.1687545776367188, -0.004347046371549368, -0.13352349400520325,
                     -0.08418072760105133, -0.5271206498146057, 0.7238689064979553,
                     -1.1414450407028198, 1.2039363384246826],
            'std': [2.377650737762451, 2.386378288269043, 2.124418020248413,
                    2.1748552322387695, 2.663944721221924, 2.371192216873169,
                    2.6217446327209473, 2.684523105621338],
        },
    )
    print(f"  数据集初始化: {time.time() - t0:.2f}s, 样本数: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    print("\n" + "=" * 60)
    print("[2/6] 加载 batch")
    print("=" * 60)
    batch = next(iter(loader))
    x_0 = batch['x_0'].to(device)
    cond_image = batch['cond'].to(device)  # (B, 3, 518, 518)
    print(f"  x_0.coords: {x_0.coords.shape}")
    print(f"  x_0.feats:  {x_0.feats.shape}")
    print(f"  x_0.shape:  {x_0.shape}")
    print(f"  cond_image: {cond_image.shape}")

    print("\n" + "=" * 60)
    print("[3/6] 初始化模型")
    print("=" * 60)
    from trellis import models

    denoiser = models.ElasticSLatFlowModel(
        resolution=64,
        in_channels=8,
        out_channels=8,
        model_channels=1024,
        cond_channels=1024,
        num_blocks=24,
        num_heads=16,
        mlp_ratio=4,
        patch_size=2,
        num_io_res_blocks=2,
        io_block_channels=[128],
        pe_mode='ape',
        qk_rms_norm=True,
        use_fp16=True,
    ).cuda()

    num_params = sum(p.numel() for p in denoiser.parameters())
    trainable = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    print(f"  模型参数量: {num_params / 1e6:.1f}M")
    print(f"  可训练参数: {trainable / 1e6:.1f}M")

    print("\n" + "=" * 60)
    print("[4/6] 加载预训练权重")
    print("=" * 60)
    t0 = time.time()
    state_dict = load_file(pretrained_path)
    missing, unexpected = denoiser.load_state_dict(state_dict, strict=False)
    print(f"  加载时间: {time.time() - t0:.2f}s")
    print(f"  missing keys:   {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")
    if missing:
        print(f"    前5个: {missing[:5]}")
    if unexpected:
        print(f"    前5个: {unexpected[:5]}")

    denoiser.train()

    print("\n" + "=" * 60)
    print("[5/6] 初始化 DINOv2 图像编码器")
    print("=" * 60)
    t0 = time.time()
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=True)
    dinov2 = dinov2.eval().cuda()
    from torchvision import transforms
    img_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f"  DINOv2 加载: {time.time() - t0:.2f}s")

    print("\n" + "=" * 60)
    print("[6/6] 执行一次训练步骤 (前向 + 反向)")
    print("=" * 60)

    # 编码条件图像
    with torch.no_grad():
        cond_normed = img_transform(cond_image)
        dinov2_out = dinov2(cond_normed, is_training=True)['x_prenorm']
        cond = F.layer_norm(dinov2_out, dinov2_out.shape[-1:])  # (B, N_patches, 1024)
    print(f"  cond encoded: {cond.shape}")

    # Flow matching 训练步骤
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=5e-5)

    noise = x_0.replace(torch.randn_like(x_0.feats))
    t = torch.rand(x_0.shape[0], device=device).float()

    # x_t = (1 - t) * x_0 + t * noise  (correct sparse broadcast)
    # Use sparse_batch_broadcast to properly handle batch dimension
    from trellis.modules.sparse.basic import sparse_batch_broadcast
    t_expanded = sparse_batch_broadcast(x_0, t.view(-1, 1))
    x_t = x_0.replace((1 - t_expanded) * x_0.feats + t_expanded * noise.feats)

    # 前向传播
    t0 = time.time()
    pred = denoiser(x_t, t * 1000, cond)
    fwd_time = time.time() - t0
    print(f"  前向传播: {fwd_time:.2f}s")
    print(f"  pred.feats: {pred.feats.shape}")

    # target = noise - x_0 (velocity)
    target = noise.replace(noise.feats - x_0.feats)

    # MSE loss
    loss = F.mse_loss(pred.feats, target.feats)
    print(f"  loss: {loss.item():.6f}")

    # 反向传播
    t0 = time.time()
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
    optimizer.step()
    bwd_time = time.time() - t0
    print(f"  反向传播: {bwd_time:.2f}s")
    print(f"  grad_norm: {grad_norm.item():.4f}")

    # 检查参数更新
    updated = 0
    for name, p in denoiser.named_parameters():
        if p.grad is not None and p.grad.abs().max() > 0:
            updated += 1
    print(f"  有梯度的参数层: {updated}/{len(list(denoiser.named_parameters()))}")

    print("\n" + "=" * 60)
    print("训练步骤验证通过 ✅")
    print("=" * 60)
    print(f"\n总结:")
    print(f"  - 数据集: {len(dataset)} 样本")
    print(f"  - 模型: ElasticSLatFlowModel ({num_params/1e6:.1f}M params)")
    print(f"  - 预训练权重: {pretrained_path}")
    print(f"  - Batch: {batch_size}, x_0.feats: {x_0.feats.shape}")
    print(f"  - Loss: {loss.item():.6f}")
    print(f"  - 前向: {fwd_time:.2f}s, 反向: {bwd_time:.2f}s")


if __name__ == '__main__':
    main()
