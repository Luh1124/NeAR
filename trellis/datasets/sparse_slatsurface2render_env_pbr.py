import os
import numpy as np
from PIL import Image
import cv2
from sympy.core.evalf import finalize_complex
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from pathlib import Path
import random
import json
import utils3d
import pyexr
import pandas as pd
import sys
from ..modules.sparse.basic import SparseTensor
from .hdri_processer import HDRI_Preprocessor
import pickle


class RelightEnvDatasetSurfacePbr(data.Dataset):
    EVEN_VIEW_RANGE = (0, 149)
    RELIGHT_VIEW_RANGE = (12, 203)
    DEFAULT_HDRI_VALUE = 0.6
    DEPTH_RANGE = (1.5, 2.5)
    DEPTH_OFFSET = 1.5
    
    def __init__(
        self,
        data_csv,
        hdri_root,
        image_size: int = 512,
        attributes_names: list = [],
        num_views: int = 40,
        grid_res: int = 64,
        num_selected_views: int = 1,
        mode: str = "train",
        max_num_voxels: int = 32768,
        random_seed: int = 42,
        even_ratio: float = 0.2,
        return_flag: str = "e",
        debug: bool = False,
    ):
        
        self.data_csv = data_csv
        self.hdri_root = Path(hdri_root)
        self.image_size = image_size
        self.attributes_names = attributes_names
        self.num_views = num_views
        self.mode = mode
        self.debug = debug
        self.max_num_voxels = max_num_voxels
        self.random_seed = random_seed
        self.even_ratio = even_ratio
        self.return_flag = return_flag

        # initialize data and preprocessor
        self.metadata = self._collect_data()
        self.hdri_preprocessor = HDRI_Preprocessor(envmap_h=512, envmap_w=1024)
        self.value_range = (0, 1)
        
        # attribute processing configuration
        self.attr_config = {
            'normal': {'channels': 3, 'ext': 'exr', 'needs_camera_transform': True},
            'depth': {'channels': 1, 'ext': 'exr', 'needs_depth_mask': True},
            'shadow': {'channels': 1, 'ext': 'exr', 'is_shadow': True},
            'Roughness': {'channels': 1, 'ext': 'exr'},
            'Metallic': {'channels': 1, 'ext': 'exr'},
            'Base Color': {'channels': 3, 'ext': 'png', 'is_base_color': True},
        }

    def _collect_data(self):
        df = pd.read_csv(self.data_csv)
        metadata = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        print(f"dataset loaded: {len(metadata)} samples (mode: {self.mode}, random seed: {self.random_seed})")
        if self.debug:
            print("sample preview:", metadata.head(5))
        
        return metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        max_tries = 10
        for i in range(max_tries):
            current_index = (index + i) % len(self)
            try:
                return self.query_item(current_index)
            except (pyexr.exr.ExrError, FileNotFoundError) as e:
                print(f"Index {current_index} data error: {e}, try next")
            except Exception as e:
                print(f"Index {current_index} unkonw error: {e}, try next")

        raise RuntimeError("Failed to retrieve a valid data item after multiple attempts")

    def query_item(self, index):
        """Query a single data item"""
        record = self.metadata.iloc[index]
        
        # 1. Determine whether to use even or relight data
        use_even_dir = random.random() < self.even_ratio
        dir_key = "even" if use_even_dir else "relight"
        
        # 2. Build paths
        image_dir = Path(record[dir_key])
        slat_path = Path(record["slat_path"])
        
        # 3. Load frame metadata
        frames_meta = self._load_frames_metadata(image_dir)
        
        # 4. Select image and HDRI
        image_meta, hdri_data = self._select_image_and_hdri(frames_meta, use_even_dir)
        
        # 5. Build camera parameters
        camera_params = self._build_camera_params(image_meta)
        
        # 6. Load image data
        image_data = self._load_image_data(image_dir, image_meta, use_even_dir)
        
        # 7. Load attribute data
        # attributes_data = self._load_attributes_data(image_dir, image_meta, use_even_dir, camera_params['c2w_mat_gl'], image_data.pop('image_gt_path'))
        attributes_data = self._load_attributes_data(image_dir, image_meta, use_even_dir, camera_params['c2w_mat_gl'], image_data.pop('image_gt_path'))
        
        # 8. Load voxel data
        voxel_data = self._load_voxel_data(slat_path)
        
        # 9. Process HDRI
        hdri_data_processed = self._process_hdri_data(hdri_data, image_meta["rotation_euler"])
        
        # 10. Merge all data
        return {"slat_path": str(slat_path), **image_data, **attributes_data, **camera_params, **voxel_data, **hdri_data_processed}

    def _load_frames_metadata(self, image_dir: Path) -> dict:
        """Load frame metadata"""
        json_path = image_dir / "transforms.json"
        with json_path.open('r') as f:
            return json.load(f)

    def _select_image_and_hdri(self, frames_meta: dict, use_even_dir: bool) -> tuple:
        """Select image and HDRI"""
        view_range = self.EVEN_VIEW_RANGE if use_even_dir else self.RELIGHT_VIEW_RANGE
        
        while True:
            # Select view index
            if self.mode in ["train", "val", "test"]:
                image_gt_idx = random.randint(*view_range)
            else:
                raise NotImplementedError(f"not supported mode: {self.mode}")
            
            image_meta = frames_meta["frames"][image_gt_idx]
            
            hdri_data = self._load_hdri_data(image_meta, use_even_dir)
            if hdri_data is not None:
                break
                
        return image_meta, hdri_data

    def _load_hdri_data(self, image_meta: dict, use_even_dir: bool) -> torch.Tensor:
        """Load HDRI data"""
        if use_even_dir:
            # even dir use default HDRI
            return torch.ones(256, 512, 3) * self.DEFAULT_HDRI_VALUE

        if not self.hdri_root.exists():
            raise FileNotFoundError(f"找不到HDRI根目录: {self.hdri_root}")
        
        # from relight json load hdri
        hdri_file_path = self.hdri_root / os.path.basename(image_meta["hdri_file_path"].replace(".exr", ".pkl"))

        if not hdri_file_path.exists():
            return None
        
        try:
            # hdri = torch.from_numpy(pyexr.read(str(hdri_file_path))[..., :3])
            hdri = torch.from_numpy(pickle.load(open(hdri_file_path, "rb"))[..., :3])
            return hdri
        except Exception as e:
            print(f"HDRI load fail {hdri_file_path}: {e}")
            return None

    def _build_camera_params(self, image_meta: dict) -> dict:
        fov = image_meta["camera_angle_x"]
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
        
        # OpenGL to OpenCV coordinate system conversion
        c2w_mat_gl = torch.tensor(image_meta["transform_matrix"])
        c2w_mat_cv = c2w_mat_gl.clone()
        c2w_mat_cv[:, 1:3] *= -1
        extrinsics = torch.inverse(c2w_mat_cv)
        
        return {
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'c2w_mat_gl': c2w_mat_gl
        }

    def _load_image_data(self, image_dir: Path, image_meta: dict, use_even_dir: bool) -> dict:
        # build file path
        image_gt_path = image_meta["file_path"]
        if use_even_dir:
            image_gt_path = image_gt_path.replace("image/", "")
            image_gt_exr_path = image_gt_path
        else:
            image_gt_exr_path = os.path.join("image", image_meta["file_path"].replace("env.png", "image_env.exr"))
        
        # load image
        image_gt_pil = Image.open(image_dir / image_gt_path)
        image_gt_exr = pyexr.read(str(image_dir / image_gt_exr_path))[..., :3].clip(min=0)

        # process alpha channel
        alpha = image_gt_pil.split()[-1] if image_gt_pil.mode == 'RGBA' else None
        
        # resize
        image_gt_exr = cv2.resize(image_gt_exr, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        if alpha is not None:
            alpha = alpha.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            alpha = torch.tensor(np.array(alpha)).float() / 255.0
        else:
            alpha = torch.ones(self.image_size, self.image_size)
        
        # convert to tensor
        image_tensor = torch.tensor(image_gt_exr).float()
        brightness_mask = self.get_brightness_mask(image_tensor)
        image_gt = image_tensor.permute(2, 0, 1)
        
        return {
            'image': image_gt,
            'alpha': alpha[None],
            'image_gt_path': image_gt_path,
            'brightness': brightness_mask[None]
        }

    def _load_attributes_data(self, image_dir: Path, image_meta: dict, use_even_dir: bool, c2w_mat_gl: torch.Tensor, image_gt_path: str) -> dict:
        attributes_dict = {}
        view_idx = str(image_meta["file_path"])[:3]
        if self.attributes_names is None:
            return {}
        for attr_name in self.attributes_names:
            attr_tensor = self._load_single_attribute(
                image_dir, attr_name, view_idx, use_even_dir, c2w_mat_gl, image_gt_path
            )
            
            # process attribute name mapping
            final_attr_name = "Basecolor" if attr_name == "Base Color" else attr_name
            attributes_dict[final_attr_name] = attr_tensor
        
        return attributes_dict

    def _load_single_attribute(self, image_dir: Path, attr_name: str, view_idx: str, use_even_dir: bool, c2w_mat_gl: torch.Tensor, image_gt_path: str) -> torch.Tensor:
        """加载单个属性"""
        config = self.attr_config.get(attr_name, {'channels': 1, 'ext': 'exr'})
        
        # build file path
        attr_path = self._build_attribute_path(image_dir, attr_name, view_idx, use_even_dir, config, image_gt_path)
        
        # load data
        if config['ext'] == 'png' and config.get('is_base_color', False):
            return self._load_base_color(attr_path)
        else:
            return self._load_exr_attribute(attr_path, attr_name, config, c2w_mat_gl)

    def _build_attribute_path(self, image_dir: Path, attr_name: str, view_idx: str, use_even_dir: bool, config: dict, image_gt_path: str) -> Path:
        # build attribute file path
        ext = config['ext']

        if config.get('is_shadow', False):
            # shadow special processing
            # filename = f"{view_idx}_{attr_name}_env.{ext}"
            filename = f"{image_gt_path.replace('env.png', 'shadow_env.exr')}"
            # if not use_even_dir:
                # filename = f"{view_idx}_000_{attr_name}_env.{ext}"
            return image_dir / "shadow" / filename
        else:
            # normal attribute
            filename = f"{view_idx}_{attr_name}.{ext}"
            if not use_even_dir:
                filename = f"{view_idx}_000_{attr_name}.{ext}"
            return image_dir / attr_name / filename

    def _load_base_color(self, attr_path: Path) -> torch.Tensor:
        # load Base Color (PNG format)
        attr_pil = Image.open(attr_path).convert('RGB')
        attr_pil = attr_pil.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        return torch.tensor(np.array(attr_pil)).permute(2, 0, 1).float() / 255.0

    def _load_exr_attribute(self, attr_path: Path, attr_name: str, config: dict, c2w_mat_gl: torch.Tensor) -> torch.Tensor:
        # load EXR format attribute
        # try to load file
        if attr_path.exists():
            attr_img = pyexr.read(str(attr_path))
        else:
            # try PNG alternative
            png_path = attr_path.with_suffix(".png")
            if png_path.exists():
                attr_img = cv2.cvtColor(cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            else:
                raise FileNotFoundError(f"Attribute file not found: {attr_path} or {png_path}")
        
        # resize and channels
        channels = config['channels']
        attr_img = cv2.resize(attr_img[..., :channels], (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        attr_tensor = torch.from_numpy(attr_img)
        
        # special processing
        if config.get('needs_camera_transform', False):
            attr_tensor = self.world2camera_normal(attr_tensor, c2w_mat_gl)
        
        if config.get('needs_depth_mask', False):
            depth_mask = (self.DEPTH_RANGE[0] < attr_tensor) & (attr_tensor < self.DEPTH_RANGE[1])
            attr_tensor = attr_tensor * depth_mask
        
        # adjust dimension
        if channels == 3:
            attr_tensor = attr_tensor.permute(2, 0, 1)
        else:
            attr_tensor = attr_tensor.unsqueeze(0)
        
        return attr_tensor

    def _load_voxel_data(self, slat_path: Path) -> dict:
        # load voxel data
        with np.load(slat_path) as slat_data:
            feats = torch.tensor(slat_data['feats']).float()
            coords = torch.tensor(slat_data['coords']).int()
            color_feats = torch.tensor(slat_data['color_feats']).float()
            brm_feats = torch.tensor(slat_data['brm_feats']).float()

        feat_length = feats.shape[0]
        if feat_length == 0:
            raise ValueError("Voxel data is empty")

        # random sampling to limit voxel number
        if feat_length > self.max_num_voxels:
            selected_idx = np.random.choice(feat_length, self.max_num_voxels, replace=False)
            feats = feats[selected_idx]
            coords = coords[selected_idx]
            color_feats = color_feats[selected_idx]
            brm_feats = brm_feats[selected_idx]
            feat_length = self.max_num_voxels

        return {
            'feats': feats,
            'coords': coords,
            'feat_length': feat_length,
            'color_feats': color_feats,
            'brm_feats': brm_feats,
        }

    def _process_hdri_data(self, hdri: torch.Tensor, hdri_rot: list) -> dict:
        # process HDRI data
        envir_map_ldr, envir_map_hdr, envir_map_perceptual, envir_map_hdr_raw, view_dirs_world = \
            self.hdri_preprocessor.preprcess_envir_map(hdri, hdri_rot)
        
        hdri_cond = torch.cat([envir_map_ldr, envir_map_hdr, view_dirs_world], dim=0).float()
        
        return {
            'hdri': hdri,
            'hdri_rot': torch.tensor(hdri_rot[2], dtype=torch.float32),
            'hdri_cond': hdri_cond,
        }

    @staticmethod
    def world2camera_normal(normal: torch.Tensor, c2w_mat_gl: torch.Tensor) -> torch.Tensor:
        # world coordinate system normal vector to camera coordinate system
        h, w, _ = normal.shape
        normal_cam = normal.reshape(-1, 3) @ c2w_mat_gl[:3, :3]
        return normal_cam.reshape(h, w, 3)

    def collate_fn(self, batch):
        # batch processing
        pack = {}
        coords = []
        
        # process coordinates
        for i, b in enumerate(batch):
            if b['coords'].shape[-1] == 3:
                batch_coords = torch.cat([
                    torch.full((b['feat_length'], 1), i, dtype=torch.int32), 
                    b['coords']
                ], dim=-1)
            elif b['coords'].shape[-1] == 4:
                batch_coords = torch.cat([
                    torch.full((b['feat_length'], 1), i, dtype=torch.int32), 
                    b['coords'][..., 1:]
                ], dim=-1)
            else:
                raise ValueError(f"Coordinate dimension error: {b['coords'].shape[-1]}, should be 3 or 4")
            coords.append(batch_coords)

        # build sparse tensor
        coords = torch.cat(coords)
        feats = torch.cat([b['feats'] for b in batch])
        color_feats = torch.cat([b['color_feats'] for b in batch])
        brm_feats = torch.cat([b['brm_feats'] for b in batch])

        if self.return_flag == "e":
            final_feats = feats
        elif self.return_flag == "b":
            final_feats = color_feats
        elif self.return_flag == "eb":
            final_feats = torch.cat([feats, color_feats], dim=1)
        elif self.return_flag == "ebr":
            final_feats = torch.cat([feats, color_feats, brm_feats], dim=1)
        elif self.return_flag == "br":
            final_feats = torch.cat([color_feats, brm_feats], dim=1)
        elif self.return_flag == "er":
            final_feats = torch.cat([feats, brm_feats], dim=1)
        else:
            raise ValueError(f"Unsupported return flag: {self.return_flag}")

        pack['latents'] = SparseTensor(
            coords=coords,
            feats=final_feats
        )
        # pack['color_latents'] = SparseTensor(
        #     coords=coords,
        #     feats=color_feats
        # )
        # pack['pbr_latents'] = SparseTensor(
        #     coords=coords,
        #     feats=brm_feats
        # )

        # process other data
        exclude_keys = {'coords', 'feats', 'color_feats', 'brm_feats', 'feat_length'}
        for key in batch[0].keys():
            if key not in exclude_keys:
                if isinstance(batch[0][key], torch.Tensor):
                    pack[key] = torch.stack([b[key] for b in batch])
                elif isinstance(batch[0][key], list):
                    pack[key] = sum([b[key] for b in batch], [])
                else:
                    pack[key] = [b[key] for b in batch]

        return pack

    def gamma_correction(self, image: torch.Tensor) -> torch.Tensor:
        # standard gamma correction
        return torch.pow(image.clamp(0, 1), 1.0 / 2.2)

    def get_brightness_mask(self, rgb_image: torch.Tensor, percentile=0.99):
        luminance_709 = (0.2126 * rgb_image[..., 0] + 
                0.7152 * rgb_image[..., 1] + 
                0.0722 * rgb_image[..., 2])
        threshold = torch.quantile(luminance_709.flatten(), percentile)
        return (luminance_709 > threshold).float()

    @torch.no_grad()
    def visualize_sample(self, sample: dict) -> dict:
        # visualize sample
        return {
            'image': self.gamma_correction(sample['image']),
            'normal': sample['normal'] * 0.5 + 0.5,
            'roughness': sample['Roughness'],
            'metallic': sample['Metallic'],
            'basecolor': sample['Basecolor'],
            'depth': sample['depth'] - self.DEPTH_OFFSET,
            'shadow': sample['shadow'],
            'brightness': sample['brightness']
        }