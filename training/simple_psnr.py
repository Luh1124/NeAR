#!/usr/bin/env python3
"""
简单的 PSNR 计算脚本，只依赖 PIL 和基础库
"""
import sys
import math
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_psnr(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    return psnr(img1, img2, data_range=255)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        img1_path = sys.argv[1]
        img2_path = sys.argv[2]
    else:
        # 默认路径
        base_dir = "/root/code/3diclight/1008_train_shade_slat_to_even_slat/outputs/dev.slat_flow_img_dit_L_64l8p2_fp16_s2e_1024_1e-5/samples/step0290000"
        img1_path = f"{base_dir}/sample_gt_step0290000.jpg"
        img2_path = f"{base_dir}/sample_step0290000.jpg"
        
    print(f"\n计算 PSNR:")
    print(f"  GT/参考图: {img1_path}")
    print(f"  生成图:    {img2_path}")
    print(f"{'-'*80}")
    
    try:
        psnr = calculate_psnr(img1_path, img2_path)
        print(f"\n结果:")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"\n")
        
        # PSNR 质量参考
        if psnr >= 40:
            quality = "优秀 (几乎无损)"
        elif psnr >= 30:
            quality = "良好"
        elif psnr >= 25:
            quality = "一般"
        elif psnr >= 20:
            quality = "较差"
        else:
            quality = "很差"
        
        print(f"质量评估: {quality}")
        print(f"{'-'*80}\n")
        
    except FileNotFoundError as e:
        print(f"\n错误: 文件未找到 - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)


