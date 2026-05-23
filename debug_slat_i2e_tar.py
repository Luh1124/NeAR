#!/usr/bin/env python3
"""
调试 / 验证 SLatI2ETar 数据集的独立脚本。
用法:
    cd /root/code/3diclight/NeAR
    python debug_slat_i2e_tar.py

功能:
    1. 数据集初始化（首次构建索引 / 加载缓存）
    2. 单样本读取测试（检查 coords/feats/cond 格式）
    3. 多 worker DataLoader 测试
    4. 随机采样一致性检查（同一 index 不同 epoch 应返回不同 view）
    5. 批量 collation 测试
    6. 性能基准测试
"""

import sys
import time
import argparse

sys.path.insert(0, '/root/code/3diclight/NeAR')
sys.path.insert(0, '/root/code/3diclight/NeAR/hy3dshape')

import torch
from torch.utils.data import DataLoader


def test_init(image_tar_dir, slat_tar_path, cache_index=True):
    """测试数据集初始化速度。"""
    print("=" * 60)
    print("[Test 1] 数据集初始化")
    print("=" * 60)

    from trellis.datasets import SLatI2ETar

    t0 = time.time()
    ds = SLatI2ETar(
        image_tar_dir=image_tar_dir,
        slat_tar_path=slat_tar_path,
        cache_index=cache_index,
    )
    t1 = time.time()

    print(f"  初始化时间: {t1 - t0:.2f}s")
    print(f"  总样本数:   {len(ds)}")
    print(f"  SLAT tar:   {slat_tar_path}")
    print(f"  Image dir:  {image_tar_dir}")
    print(f"  前 10 个样本的 num_voxels: {ds.loads[:10]}")
    print(f"  max_voxels: {max(ds.loads)}")
    print(f"  min_voxels: {min(ds.loads)}")
    print(f"  avg_voxels: {sum(ds.loads) / len(ds.loads):.1f}")
    return ds


def test_single_sample(ds, num_samples=5):
    """测试单样本读取。"""
    print("\n" + "=" * 60)
    print("[Test 2] 单样本读取")
    print("=" * 60)

    for i in range(num_samples):
        idx = (i * len(ds) // num_samples) % len(ds)
        t0 = time.time()
        sample = ds[idx]
        t1 = time.time()

        print(f"\n  Sample [{idx}] (time: {t1 - t0:.3f}s):")
        print(f"    coords:      {sample['coords'].shape}  dtype={sample['coords'].dtype}")
        print(f"    feats:       {sample['feats'].shape}  dtype={sample['feats'].dtype}")
        print(f"    cond:        {sample['cond'].shape}  dtype={sample['cond'].dtype}")
        print(f"    cond range:  [{sample['cond'].min():.3f}, {sample['cond'].max():.3f}]")


def test_multi_view_consistency(ds, idx=0, num_trials=5):
    """验证同一 index 在不同调用时返回不同 view（随机采样）。"""
    print("\n" + "=" * 60)
    print("[Test 3] 多视角随机采样一致性")
    print("=" * 60)
    print(f"  对 index={idx} 连续采样 {num_trials} 次，cond 应该不同：")

    conds = []
    for t in range(num_trials):
        sample = ds[idx]
        cond = sample['cond']
        cond_hash = torch.sum(cond).item()
        conds.append(cond_hash)
        print(f"    Trial {t}: cond sum={cond_hash:.4f}")

    unique = len(set(conds))
    if unique == 1:
        print(f"  ⚠️ 警告: 所有 trial 的 cond 完全相同（可能只有一个 view）")
    else:
        print(f"  ✅ {unique}/{num_trials} 次采样返回了不同的 view")


def test_dataloader(ds, batch_size=4, num_workers=2, num_batches=3):
    """测试多 worker DataLoader。"""
    print("\n" + "=" * 60)
    print("[Test 4] DataLoader 多 worker 测试")
    print("=" * 60)
    print(f"  batch_size={batch_size}, num_workers={num_workers}")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=ds.collate_fn,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        t1 = time.time()
        x0 = batch['x_0']
        print(f"\n  Batch {i} (iter_time: {t1 - t0:.3f}s):")
        print(f"    x_0.coords:  {x0.coords.shape}")
        print(f"    x_0.feats:   {x0.feats.shape}")
        print(f"    cond:        {batch['cond'].shape}")
        print(f"    layout:      {len(x0.layout)} slices")
        t0 = time.time()


def test_collation_split(ds, batch_size=8, split_size=None):
    """测试不同 batch size 和 split_size 的 collation。"""
    print("\n" + "=" * 60)
    print("[Test 5] Collation 测试")
    print("=" * 60)

    indices = list(range(min(batch_size, len(ds))))
    raw_batch = [ds[i] for i in indices]

    # 不分组
    pack = ds.collate_fn(raw_batch, split_size=None)
    print(f"  split_size=None:")
    print(f"    x_0.coords: {pack['x_0'].coords.shape}")
    print(f"    cond:       {pack['cond'].shape}")

    # 按 voxel 数分组
    if split_size:
        packs = ds.collate_fn(raw_batch, split_size=split_size)
        print(f"\n  split_size={split_size}:")
        for j, p in enumerate(packs):
            print(f"    Pack {j}: coords={p['x_0'].coords.shape}, cond={p['cond'].shape}")


def benchmark_throughput(ds, batch_size=8, num_workers=4, num_batches=20):
    """性能基准测试。"""
    print("\n" + "=" * 60)
    print("[Test 6] 吞吐量基准测试")
    print("=" * 60)
    print(f"  batch_size={batch_size}, num_workers={num_workers}, batches={num_batches}")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=ds.collate_fn,
        prefetch_factor=2,
        persistent_workers=True,
    )

    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        # 模拟一个轻量 GPU 操作，避免 CPU 空转
        _ = batch['cond'] * 1.0

    elapsed = time.time() - t0
    total_samples = min(num_batches, i + 1) * batch_size
    print(f"\n  总时间:     {elapsed:.2f}s")
    print(f"  总样本:     {total_samples}")
    print(f"  吞吐量:     {total_samples / elapsed:.1f} samples/s")
    print(f"  每 batch:   {elapsed / min(num_batches, i + 1):.3f}s")


def main():
    parser = argparse.ArgumentParser(description="调试 SLatI2ETar 数据集")
    parser.add_argument('--image-tar-dir', default='/root/data/3diclight/downloaded_ref_images/ref_images')
    parser.add_argument('--slat-tar-path', default='/root/data/3diclight/LH-Slat.tar')
    parser.add_argument('--cache-index', action='store_true', default=True)
    parser.add_argument('--no-cache-index', dest='cache_index', action='store_false')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--skip-throughput', action='store_true', help='跳过耗时吞吐量测试')
    args = parser.parse_args()

    # 1. 初始化
    ds = test_init(args.image_tar_dir, args.slat_tar_path, args.cache_index)

    # 2. 单样本
    test_single_sample(ds, num_samples=5)

    # 3. 多视角一致性
    test_multi_view_consistency(ds, idx=0, num_trials=5)

    # 4. DataLoader
    test_dataloader(ds, batch_size=args.batch_size, num_workers=args.num_workers, num_batches=3)

    # 5. Collation
    test_collation_split(ds, batch_size=8, split_size=2)

    # 6. 吞吐量
    if not args.skip_throughput:
        benchmark_throughput(ds, batch_size=args.batch_size, num_workers=args.num_workers, num_batches=20)

    print("\n" + "=" * 60)
    print("所有测试通过 ✅")
    print("=" * 60)


if __name__ == '__main__':
    main()
