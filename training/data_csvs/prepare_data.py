import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_single_sha256(sha256, slat_dir, shade_dir, relight_dir, n_shade=12):
    """
    单个 sha256 的文件检查，返回所有有效的 (even_slat, shade_slat, relight_image) 三元组
    
    Returns:
        pairs: List[Tuple[str, str, str]] - (even_slat_path, shade_slat_path, relight_image_path)
        missing_slats: int - 缺失的 even slat 数量 (0 or 1)
        missing_shades: int - 缺失的 shade slat 数量
    """
    slat_path = os.path.join(slat_dir, f"{sha256}.npz")
    shade_slat_dir = os.path.join(shade_dir, sha256)
    relight_image_dir = os.path.join(relight_dir, sha256)
    
    # 检查 even slat 是否存在
    if not os.path.exists(slat_path):
        return [], 1, 0

    if not os.path.exists(shade_slat_dir):
        return [], 0, 1

    if not os.path.exists(relight_image_dir):
        return [], 0, 1
    
    pairs = []
    missing_shades = 0
    
    for i in range(n_shade):
        shade_slat_path = os.path.join(shade_slat_dir, f"{i:03d}_000_random_area.npz")
        num_voxel = np.load(shade_slat_path)['coords'].shape[0]
        relight_image_path = os.path.join(relight_image_dir, f"{i:03d}_000_random_area.png")
        
        # 同时检查 shade slat 和 relight image 是否存在
        if os.path.exists(shade_slat_path) and os.path.exists(relight_image_path):
            pairs.append((slat_path, shade_slat_path, relight_image_path, num_voxel))
        else:
            missing_shades += 1
            # 可选：更详细的日志
            # if not os.path.exists(shade_slat_path):
            #     print(f"Missing shade slat: {shade_slat_path}")
            # if not os.path.exists(relight_image_path):
            #     print(f"Missing relight image: {relight_image_path}")
    
    return pairs, 0, missing_shades


def collect_pairs_parallel(sha256s, slat_dir, shade_dir, relight_dir, n_shade=12, 
                          max_workers=16, verbose=True, desc=''):
    """
    并行收集所有有效的 (even_slat, shade_slat, relight_image) 三元组
    
    Returns:
        even_slats: List[str] - even slat 文件路径
        shade_slats: List[str] - shade slat 文件路径
        relight_paths: List[str] - relight 图像路径
        num_voxels: List[int] - 每个样本的 voxel 数量
    """
    even_slats = []
    shade_slats = []
    relight_paths = []
    num_voxels = []

    missing_slats = 0
    missing_shades = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_sha = {
            executor.submit(check_single_sha256, sha256, slat_dir, shade_dir, relight_dir, n_shade): sha256
            for sha256 in sha256s
        }
        
        # 使用 tqdm 显示进度
        if verbose:
            futures = tqdm(as_completed(future_to_sha), total=len(sha256s), desc=desc)
        else:
            futures = as_completed(future_to_sha)
        
        for future in futures:
            sha256 = future_to_sha[future]
            try:
                pairs, miss_slat, miss_shade = future.result()
                missing_slats += miss_slat
                missing_shades += miss_shade
                
                # 解包三元组
                for even, shade, relight, num_voxel in pairs:
                    even_slats.append(even)
                    shade_slats.append(shade)
                    relight_paths.append(relight)
                    num_voxels.append(num_voxel)
                
                # 警告：有 even slat 但没有任何有效 shade
                if not pairs and miss_slat == 0 and verbose:
                    tqdm.write(f"[WARN] {sha256}: has even slat but no valid shade pairs")
                    
            except Exception as e:
                if verbose:
                    tqdm.write(f"[ERROR] Processing {sha256}: {e}")
    
    if verbose:
        print(f"\n  Summary for {desc}:")
        print(f"    Valid pairs: {len(even_slats)}")
        print(f"    Missing even slats: {missing_slats}")
        print(f"    Missing shade slats: {missing_shades}")
    
    return even_slats, shade_slats, relight_paths, num_voxels


def main(
    data_csv_3w9,
    data_csv_4w8,
    val_data_csv_3w9,
    val_data_csv_4w8,
    slat_dir_3w9,
    slat_dir_4w8,
    shade_dir,
    relight_dir,
    n_shade=12,
    max_workers=16,
    out_train="train_shade_slat_to_even_slat.csv",
    out_val="val_shade_slat_to_even_slat.csv",
    verbose=True
):
    """
    主函数：收集训练和验证数据
    
    输出 CSV 格式：
        even_slats, shade_slats, relight_images
    """
    
    # 读取 sha256 列表
    print("Loading sha256 lists...")
    sha256s_3w9 = sorted(pd.read_csv(data_csv_3w9)['sha256'].tolist())
    sha256s_4w8 = sorted(pd.read_csv(data_csv_4w8)['sha256'].tolist())
    val_sha256s_3w9 = sorted(pd.read_csv(val_data_csv_3w9)['sha256'].tolist())
    val_sha256s_4w8 = sorted(pd.read_csv(val_data_csv_4w8)['sha256'].tolist())
    
    print(f"  Train 3w9: {len(sha256s_3w9)} samples")
    print(f"  Train 4w8: {len(sha256s_4w8)} samples")
    print(f"  Val 3w9: {len(val_sha256s_3w9)} samples")
    print(f"  Val 4w8: {len(val_sha256s_4w8)} samples")
    
    # ===== 训练集 =====
    print("\n" + "="*60)
    print("Collecting TRAINING pairs (parallel)...")
    print("="*60)
    
    even1, shade1, relight1, num_voxels1 = collect_pairs_parallel(
        sha256s_3w9, slat_dir_3w9, shade_dir, relight_dir, 
        n_shade, max_workers, verbose, desc='TRAIN 3w9'
    )
    
    even2, shade2, relight2, num_voxels2 = collect_pairs_parallel(
        sha256s_4w8, slat_dir_4w8, shade_dir, relight_dir, 
        n_shade, max_workers, verbose, desc='TRAIN 4w8'
    )
    
    # 合并
    even_slats = even1 + even2
    shade_slats = shade1 + shade2
    relight_images = relight1 + relight2
    num_voxels = num_voxels1 + num_voxels2

    # 保存训练集 CSV
    df_train = pd.DataFrame({
        "even_slat_path": even_slats,
        "shade_slat_path": shade_slats,
        "relight_image_path": relight_images,
        "num_voxels": num_voxels
    })
    df_train.to_csv(out_train, index=False)
    print(f"\n✓ Saved: {out_train}")
    print(f"  Total pairs: {len(df_train)}")
    
    # ===== 验证集 =====
    print("\n" + "="*60)
    print("Collecting VALIDATION pairs (parallel)...")
    print("="*60)
    
    val_even1, val_shade1, val_relight1, val_num_voxels1 = collect_pairs_parallel(
        val_sha256s_3w9, slat_dir_3w9, shade_dir, relight_dir, 
        n_shade, max_workers, verbose, desc='VAL 3w9'
    )
    
    val_even2, val_shade2, val_relight2, val_num_voxels2 = collect_pairs_parallel(
        val_sha256s_4w8, slat_dir_4w8, shade_dir, relight_dir, 
        n_shade, max_workers, verbose, desc='VAL 4w8'
    )
    
    # 合并
    val_even_slats = val_even1 + val_even2
    val_shade_slats = val_shade1 + val_shade2
    val_relight_images = val_relight1 + val_relight2
    val_num_voxels = val_num_voxels1 + val_num_voxels2

    # 保存验证集 CSV
    df_val = pd.DataFrame({
        "even_slat_path": val_even_slats,
        "shade_slat_path": val_shade_slats,
        "relight_imageh_path": val_relight_images,
        "num_voxels": val_num_voxels
    })
    df_val.to_csv(out_val, index=False)
    print(f"\n✓ Saved: {out_val}")
    print(f"  Total pairs: {len(df_val)}")
    
    # ===== 最终统计 =====
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print(f"Training set:   {len(df_train):,} pairs")
    print(f"Validation set: {len(df_val):,} pairs")
    print(f"Total:          {len(df_train) + len(df_val):,} pairs")
    print("="*60)


if __name__ == "__main__":
    main(
        # 训练集元数据
        data_csv_3w9="data_csvs/train_3w9_0518.csv",
        data_csv_4w8="data_csvs/train_4w8.csv",
        
        # 验证集元数据
        val_data_csv_3w9="data_csvs/val_3w9_0518.csv",
        val_data_csv_4w8="data_csvs/val_4w8.csv",
        
        # 数据目录
        slat_dir_3w9="/root/data/3diclight/pbr2dino_voxel_latent/3w9",
        slat_dir_4w8="/root/data/3diclight/pbr2dino_voxel_latent/5w8",
        # shade_dir="/root/data/3diclight/image2slat_trellis_flow_model_rerun",
        shade_dir="/root/data/3diclight/image2slat_trellis_flow_model_rerun_v3",
        # relight_dir="/root/data/3diclight/8w9_neural_light_v6",
        relight_dir="/root/data/3diclight/8w9_neural_light_v6",
        
        # 参数
        n_shade=12,           # 每个样本的 shade 数量
        max_workers=32,       # 并行线程数（可根据机器调整 8-32）
        
        # 输出文件
        out_train="train_shade_slat_to_even_slat_v3.csv",
        out_val="val_shade_slat_to_even_slat_v3.csv",
        
        verbose=True
    )