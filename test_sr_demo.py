import os
import sys
import time
import torch
import numpy as np
from PIL import Image

# Add DSCF-SR to Python path
sys.path.insert(0, os.path.abspath("./DSCF-SR"))
from models.team23_DSCF import DSCF

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize DSCF Super-Resolution model (4x upscale)
    print("Initializing DSCF model...")
    model = DSCF(num_in_ch=3, num_out_ch=3, feature_channels=26, upscale=4)
    
    # 2. Load pre-trained weights
    model_path = "./DSCF-SR/model_zoo/team23_DSCF.pth"
    if not os.path.exists(model_path):
        print(f"Error: Weight file not found at {model_path}")
        return
        
    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)
    
    # 3. Load test image
    test_image_path = "assets/example_image/typical_misc_mailbox.png"
    if not os.path.exists(test_image_path):
        # Fallback to any PNG in example gallery
        import glob
        pngs = glob.glob("assets/example_image/*.png")
        if pngs:
            test_image_path = pngs[0]
        else:
            print("Error: No test images found in assets/example_image/")
            return
            
    print(f"Loading test image: {test_image_path}")
    orig_img = Image.open(test_image_path).convert("RGB")
    orig_w, orig_h = orig_img.size
    print(f"Original resolution: {orig_w}x{orig_h}")
    
    # Crop to multiple of 4 for clean comparison
    crop_w = (orig_w // 4) * 4
    crop_h = (orig_h // 4) * 4
    orig_img = orig_img.crop((0, 0, crop_w, crop_h))
    print(f"Cropped to multiple of 4: {crop_w}x{crop_h}")
    
    # 4. Generate low-resolution image by 4x downsampling (acting as LQ input)
    lq_w = crop_w // 4
    lq_h = crop_h // 4
    print(f"Creating low-res image by 4x downsampling to: {lq_w}x{lq_h}")
    lq_img = orig_img.resize((lq_w, lq_h), Image.Resampling.BILINEAR)
    
    # 5. Create standard bilinear-upsampled baseline for comparison
    bilinear_baseline = lq_img.resize((crop_w, crop_h), Image.Resampling.BILINEAR)
    
    # 6. Run DSCF Super-Resolution model on GPU
    # Preprocess image to tensor [0, 1] range and send to GPU
    lq_np = np.array(lq_img) # HxWxC uint8
    lq_tensor = torch.from_numpy(lq_np.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
    
    print("Running DSCF super-resolution model...")
    t0 = time.time()
    with torch.no_grad():
        # Warmup
        _ = model(lq_tensor)
        torch.cuda.synchronize()
        
        # Benchmark timing
        t_start = time.time()
        sr_tensor = model(lq_tensor)
        torch.cuda.synchronize()
        t_end = time.time()
        
    inference_time_ms = (t_end - t_start) * 1000
    print(f"Super-resolution completed in {inference_time_ms:.2f} ms!")
    
    # Postprocess tensor back to PIL Image
    sr_tensor = sr_tensor.clamp(0.0, 1.0).squeeze(0).cpu()
    sr_np = (sr_tensor.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    sr_img = Image.fromarray(sr_np)
    
    # 7. Create output folder and save results
    output_dir = "./tmp_sr_test"
    os.makedirs(output_dir, exist_ok=True)
    
    orig_img.save(f"{output_dir}/1_original.png")
    lq_img.save(f"{output_dir}/2_low_res.png")
    bilinear_baseline.save(f"{output_dir}/3_bilinear_upsampled.png")
    sr_img.save(f"{output_dir}/4_dscf_super_resolved.png")
    
    print("\n" + "="*50)
    print("🎉 Super-Resolution Test Completed successfully!")
    print(f"Results saved in: {os.path.abspath(output_dir)}")
    print(f"  - 1_original.png: Original high-res ({crop_w}x{crop_h})")
    print(f"  - 2_low_res.png: Downsampled low-res ({lq_w}x{lq_h})")
    print(f"  - 3_bilinear_upsampled.png: Simple bilinear interpolation ({crop_w}x{crop_h})")
    print(f"  - 4_dscf_super_resolved.png: DSCF AI super-resolved ({crop_w}x{crop_h})")
    print(f"Inference speed: {inference_time_ms:.2f} ms")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
