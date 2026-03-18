import os
import sys

import argparse
import time
import imageio
import numpy as np
from PIL import Image

sys.path.insert(0, './hy3dshape')
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

from trellis.pipelines import NeARImageToRelightable3DPipeline
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline


def main():
    parser = argparse.ArgumentParser(description='NeAR: from image or SLaT to relightable 3D')
    parser.add_argument('--checkpoint', type=str, default='checkpoints',
                        help='Pipeline weights directory (contains pipeline.yaml)')
    parser.add_argument('--image', type=str, default="assets/example_image/T.png",
                        help='Input image path (one of --slat or --image)')
    parser.add_argument('--slat', type=str, default=None,
                        help='SLaT .npz path (one of --image or --slat)')
    parser.add_argument('--hdri', type=str, default="assets/hdris/studio_small_03_1k.exr",
                        help='HDRI .exr path')
    parser.add_argument('--out_dir', type=str, default='relight_out',
                        help='Output directory')
    parser.add_argument('--yaw', type=float, default=0.0, help='View yaw (degrees) ')
    parser.add_argument('--pitch', type=float, default=0.0, help='View pitch (degrees)')
    parser.add_argument('--hdri_rot', type=float, default=0.0,
                        help='HDRI rotation angle (degrees)')
    parser.add_argument('--video_frames', type=int, default=40,
                        help='Render additional spiral camera path video frames')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_slat', type=str, default=None,
                        help='When generating from image, save SLaT to this .npz path')
    parser.add_argument('--no_cuda', action='store_true', help='Do not use CUDA')
    args = parser.parse_args()

    if (args.image is None) == (args.slat is None):
        parser.error('Please specify --image or --slat, one of them')
    from_image = args.image is not None

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if not args.no_cuda else 'cpu'

    total_t0 = time.perf_counter()
    t0 = time.perf_counter()
    hyshape_model_id = 'tencent/Hunyuan3D-2.1'
    hyshape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(hyshape_model_id)
    hyshape_pipe.to(device)
    pipeline = NeARImageToRelightable3DPipeline.from_pretrained(args.checkpoint)
    pipeline.to(device)
    print(f"  [OK] Pipeline loaded, +{time.perf_counter() - t0:.1f}s")

    if from_image:
        image = Image.open(args.image).convert('RGB')
        image_prep = pipeline.preprocess_image(image)

        t0 = time.perf_counter()
        mesh = hyshape_pipe(image=image_prep)[0]
        mesh_path = os.path.join(args.out_dir, 'initial_3d_shape.glb')
        mesh.export(mesh_path)
        print(f"  [OK] Geometry mesh generated, +{time.perf_counter() - t0:.1f}s, saved to {mesh_path}")

        t0 = time.perf_counter()
        slat = pipeline.run_with_shape(
            image_prep,
            mesh,
            seed=args.seed,
            preprocess_image=False,
        )
        print(f"  [OK] Image → SLaT generated, +{time.perf_counter() - t0:.1f}s")
        if args.save_slat:
            np.savez(
                args.save_slat,
                feats=slat.feats.cpu().numpy(),
                coords=slat.coords.cpu().numpy(),
            )
            print(f"  [OK] Saved SLaT to: {args.save_slat}")
    else:
        t0 = time.perf_counter()
        slat = pipeline.load_slat(args.slat)
        print(f"  [OK] Loaded SLaT, +{time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    hdri_np = pipeline.load_hdri(args.hdri)
    print(f"  [OK] Loaded HDRI, +{time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()

    pipeline.renderer.ssaa = 1
    pipeline.renderer.resolution = 1024

    views = pipeline.render_view(
        slat, hdri_np,
        yaw_deg=args.yaw, pitch_deg=args.pitch,
        fov=40.0, radius=2.0, hdri_rot_deg=args.hdri_rot,
        resolution=512
    )

    color_path = os.path.join(args.out_dir, 'relight_color.png')
    base_color_path = os.path.join(args.out_dir, 'base_color.png')
    metallic_path = os.path.join(args.out_dir, 'metallic.png')
    roughness_path = os.path.join(args.out_dir, 'roughness.png')
    shadow_path = os.path.join(args.out_dir, 'shadow.png')
    views['color'].save(color_path)
    views['base_color'].save(base_color_path)
    views['metallic'].save(metallic_path)
    views['roughness'].save(roughness_path)
    views['shadow'].save(shadow_path)

    print(f"  [OK] Single view rendering completed, +{time.perf_counter() - t0:.1f}s, saved to {color_path}, {base_color_path}, {metallic_path}, {roughness_path}, {shadow_path}")

    if args.video_frames > 0:
        t0 = time.perf_counter()
        frames = pipeline.render_camera_path_video(
            slat, hdri_np, num_views=args.video_frames,
            fov=40.0, radius=2.0, hdri_rot_deg=args.hdri_rot,
            verbose=True, full_video=True, shadow_video=True
        )
        video_path = os.path.join(args.out_dir, 'relight_camera_path.mp4')
        imageio.mimsave(video_path, frames, fps=24)
        print(f"  [OK] Camera path video completed, +{time.perf_counter() - t0:.1f}s, saved to {video_path}")
    print("Done.")

    if args.video_frames > 0:
        t0 = time.perf_counter()
        hdri_roll_frames, render_frames = pipeline.render_hdri_rotation_video(
            slat, hdri_np, num_frames=args.video_frames,
            yaw_deg=args.yaw, pitch_deg=args.pitch,
            fov=40.0, radius=2.0,
            verbose=True, full_video=True, shadow_video=True
        )
        hdri_video_path = os.path.join(args.out_dir, 'hdri_roll.mp4')
        render_video_path = os.path.join(args.out_dir, 'relight_hdri_rotation.mp4')
        imageio.mimsave(hdri_video_path, hdri_roll_frames, fps=24)
        imageio.mimsave(render_video_path, render_frames, fps=24)
        print(
            f"  [OK] HDRI rotation videos completed, +{time.perf_counter() - t0:.1f}s, "
            f"saved to {hdri_video_path} and {render_video_path}"
        )

    print(f"  [OK] Total time: {time.perf_counter() - total_t0:.1f}s")


if __name__ == '__main__':
    main()
