#!/usr/bin/env python3
"""
计算两张图片之间的 PSNR
"""
import os
import argparse
import numpy as np
from PIL import Image
import cv2


def calculate_psnr(img1_path, img2_path):
    """
    计算两张图片之间的 PSNR
    
    Args:
        img1_path: 第一张图片路径（真实图/参考图）
        img2_path: 第二张图片路径（生成图/待评估图）
    
    Returns:
        psnr: PSNR 值
    """
    # 读取图片
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    
    # 确保两张图片大小一致
    if img1.size != img2.size:
        print(f"警告: 图片尺寸不一致！")
        print(f"  img1: {img1.size}")
        print(f"  img2: {img2.size}")
        print(f"  将 img2 resize 到 img1 的尺寸")
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
    
    # 转换为 numpy 数组
    img1_array = np.array(img1).astype(np.float64)
    img2_array = np.array(img2).astype(np.float64)
    
    # 计算 MSE
    mse = np.mean((img1_array - img2_array) ** 2)
    
    if mse == 0:
        return float('inf')  # 两张图片完全相同
    
    # 计算 PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr


def calculate_psnr_cv2(img1_path, img2_path):
    """
    使用 OpenCV 计算 PSNR（作为验证）
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    psnr = cv2.PSNR(img1, img2)
    return psnr


def calculate_batch_psnr(dir_path, pattern1='sample_gt_*.jpg', pattern2='sample_step*.jpg'):
    """
    批量计算目录下所有匹配的图片对的 PSNR
    """
    import glob
    
    gt_files = sorted(glob.glob(os.path.join(dir_path, pattern1)))
    sample_files = sorted(glob.glob(os.path.join(dir_path, pattern2)))
    
    if not gt_files or not sample_files:
        # 尝试其他可能的模式
        gt_files = sorted(glob.glob(os.path.join(dir_path, '**', pattern1), recursive=True))
        sample_files = sorted(glob.glob(os.path.join(dir_path, '**', pattern2), recursive=True))
    
    print(f"\n找到 {len(gt_files)} 个 GT 图片")
    print(f"找到 {len(sample_files)} 个 Sample 图片\n")
    
    psnr_list = []
    for gt_file in gt_files:
        basename = os.path.basename(gt_file)
        step = basename.split('_')[-1].split('.')[0]  # 提取 step 编号
        
        # 找到对应的 sample 文件
        sample_file = None
        for sf in sample_files:
            if step in sf and 'sample_step' in sf and 'gt' not in sf and 'shaded' not in sf:
                sample_file = sf
                break
        
        if sample_file:
            psnr = calculate_psnr(gt_file, sample_file)
            psnr_list.append(psnr)
            print(f"[{step}] PSNR: {psnr:.2f} dB")
            print(f"  GT:     {os.path.basename(gt_file)}")
            print(f"  Sample: {os.path.basename(sample_file)}")
    
    if psnr_list:
        print(f"\n{'='*60}")
        print(f"平均 PSNR: {np.mean(psnr_list):.2f} dB")
        print(f"最小 PSNR: {np.min(psnr_list):.2f} dB")
        print(f"最大 PSNR: {np.max(psnr_list):.2f} dB")
        print(f"标准差:    {np.std(psnr_list):.2f} dB")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='计算两张图片之间的 PSNR')
    parser.add_argument('--img1', type=str, help='第一张图片路径（GT/参考图）')
    parser.add_argument('--img2', type=str, help='第二张图片路径（生成图）')
    parser.add_argument('--dir', type=str, help='目录路径，批量计算该目录下所有图片对的 PSNR')
    parser.add_argument('--use-cv2', action='store_true', help='使用 OpenCV 计算（需要安装 opencv-python）')
    
    args = parser.parse_args()
    
    if args.dir:
        # 批量计算
        calculate_batch_psnr(args.dir)
    elif args.img1 and args.img2:
        # 单张图片计算
        if args.use_cv2:
            try:
                psnr = calculate_psnr_cv2(args.img1, args.img2)
                print(f"PSNR (OpenCV): {psnr:.2f} dB")
            except Exception as e:
                print(f"OpenCV 计算失败: {e}")
                print("使用 NumPy 方法...")
                psnr = calculate_psnr(args.img1, args.img2)
                print(f"PSNR (NumPy): {psnr:.2f} dB")
        else:
            psnr = calculate_psnr(args.img1, args.img2)
            print(f"\nPSNR: {psnr:.2f} dB")
            print(f"  GT/参考图: {args.img1}")
            print(f"  生成图:    {args.img2}\n")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

