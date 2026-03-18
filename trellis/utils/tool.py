import numpy as np
from PIL import Image
import trimesh
import pymeshlab

class imageSuperNet:
    def __init__(self, model_path) -> None:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            gpu_id=None,
        )
        self.upsampler = upsampler

    def __call__(self, image):
        if isinstance(image, Image.Image):
            output, _ = self.upsampler.enhance(np.array(image))
            output = Image.fromarray(output)
        elif isinstance(image, np.ndarray):
            output, _ = self.upsampler.enhance(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        return output

def remesh_mesh(mesh_path, remesh_path):
    mesh = mesh_simplify_trimesh(mesh_path, remesh_path)
    return mesh


def mesh_simplify_trimesh(inputpath, outputpath, target_count=40000):
    # 先去除离散面
    ms = pymeshlab.MeshSet()
    if inputpath.endswith(".glb"):
        ms.load_new_mesh(inputpath, load_in_a_single_layer=True)
    else:
        ms.load_new_mesh(inputpath)
    ms.save_current_mesh(outputpath.replace(".glb", ".obj"), save_textures=False)
    # 调用减面函数
    courent = trimesh.load(outputpath.replace(".glb", ".obj"), force="mesh")
    face_num = courent.faces.shape[0]

    if face_num > target_count:
        courent = courent.simplify_quadric_decimation(target_count)
    courent.export(outputpath)
    return courent
