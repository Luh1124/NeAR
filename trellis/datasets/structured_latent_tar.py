import io
import json
import os
import struct
from typing import *

import numpy as np
from PIL import Image
import torch

from ..modules.sparse.basic import SparseTensor
from ..utils.data_utils import load_balanced_group_indices
from .structured_latent import SLatVisMixin


_TAR_HEADER_SIZE = 512


def _scan_tar_fast(tar_path: str):
    """Scan a tar file and return a list of (name, offset_data, size) for regular files.
    Uses raw os.read to avoid Python tarfile object overhead (~100x faster for large tars)."""
    fd = os.open(tar_path, os.O_RDONLY)
    try:
        members = []
        offset = 0
        while True:
            header = os.read(fd, _TAR_HEADER_SIZE)
            if len(header) < _TAR_HEADER_SIZE:
                break

            # End-of-archive: two zero blocks
            if header.count(b'\x00') == _TAR_HEADER_SIZE:
                next_block = os.read(fd, _TAR_HEADER_SIZE)
                if len(next_block) == _TAR_HEADER_SIZE and next_block.count(b'\x00') == _TAR_HEADER_SIZE:
                    break
                else:
                    os.lseek(fd, -_TAR_HEADER_SIZE, os.SEEK_CUR)
                    offset += _TAR_HEADER_SIZE
                    continue

            name = header[:100].split(b'\x00')[0].decode('utf-8', errors='replace')
            size_str = header[124:136].split(b'\x00')[0].decode('ascii', errors='replace')
            try:
                size = int(size_str, 8) if size_str else 0
            except ValueError:
                size = 0
            typeflag = chr(header[156]) if len(header) > 156 else '\x00'

            if typeflag in ('0', '\x00', '') and size > 0:
                members.append((name, offset + _TAR_HEADER_SIZE, size))

            padded_size = ((size + 511) // 512) * 512
            os.lseek(fd, padded_size, os.SEEK_CUR)
            offset += _TAR_HEADER_SIZE + padded_size
        return members
    finally:
        os.close(fd)


def _get_coords_count_from_npz(fd: int, offset: int, size: int) -> int:
    """Extract coords voxel count from an npz stored inside a tar, by reading only the zip central directory.
    ~100x faster than full np.load. Assumes coords dtype is uint8 and header size is 128."""
    # Read last 4KB (enough to contain EOCD for typical npz files)
    tail_size = min(size, 4096)
    os.lseek(fd, offset + size - tail_size, os.SEEK_SET)
    tail = os.read(fd, tail_size)

    eocd_pos = tail.rfind(b'\x50\x4b\x05\x06')
    if eocd_pos < 0:
        # Fallback: full read
        os.lseek(fd, offset, os.SEEK_SET)
        npz = np.load(io.BytesIO(os.read(fd, size)))
        return int(npz['coords'].shape[0])

    eocd_abs = offset + size - tail_size + eocd_pos
    os.lseek(fd, eocd_abs, os.SEEK_SET)
    eocd = os.read(fd, 22)
    cd_offset = struct.unpack('<I', eocd[16:20])[0]
    cd_size = struct.unpack('<I', eocd[12:16])[0]

    os.lseek(fd, offset + cd_offset, os.SEEK_SET)
    cd = os.read(fd, cd_size)

    pos = 0
    while pos < len(cd):
        sig = struct.unpack('<I', cd[pos:pos + 4])[0]
        if sig != 0x02014b50:
            break
        uncomp_size = struct.unpack('<I', cd[pos + 24:pos + 28])[0]
        fname_len = struct.unpack('<H', cd[pos + 28:pos + 30])[0]
        extra_len = struct.unpack('<H', cd[pos + 30:pos + 32])[0]
        comment_len = struct.unpack('<H', cd[pos + 32:pos + 34])[0]
        fname = cd[pos + 46:pos + 46 + fname_len].decode('utf-8')
        if fname == 'coords.npy':
            return (uncomp_size - 128) // 3
        pos += 46 + fname_len + extra_len + comment_len

    # Fallback
    os.lseek(fd, offset, os.SEEK_SET)
    npz = np.load(io.BytesIO(os.read(fd, size)))
    return int(npz['coords'].shape[0])


class SLatI2ETar(SLatVisMixin):
    """
    Image-conditioned structured latent dataset for I2E training,
    reading images from WebDataset shards and latents from a single tar archive.

    Optimized for large (70GB+) tar files using raw offset-based reading:
    - Tar header scanning via os.read (~100x faster than tarfile module)
    - NumPy voxel count extraction via zip central directory parsing (~100x faster than np.load)
    """

    def __init__(
        self,
        image_tar_dir: str,
        slat_tar_path: str,
        *,
        max_num_voxels: int = 32768,
        image_size: int = 518,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        cache_index: bool = True,
    ):
        self.image_tar_dir = image_tar_dir
        self.slat_tar_path = slat_tar_path
        self.max_num_voxels = max_num_voxels
        self.image_size = image_size
        self.value_range = (0, 1)
        self.normalization = normalization

        super().__init__(
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
        )

        # Build or load index
        index_path = slat_tar_path + '.index.json' if cache_index else None
        if index_path and os.path.exists(index_path):
            self._load_index(index_path)
        else:
            self._build_index()
            if index_path:
                self._save_index(index_path)

        # Filter by max_num_voxels
        self.samples = [s for s in self.samples if s['num_voxels'] <= self.max_num_voxels]
        if len(self.samples) == 0:
            raise RuntimeError("No samples left after filtering by max_num_voxels")

        self.loads = [s['num_voxels'] for s in self.samples]

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1)

        # Open raw file descriptors for fast offset-based reading.
        # Each DataLoader worker gets its own copy of the dataset,
        # so each worker has independent FDs.
        self._image_fds: Dict[str, int] = {}
        for name, path in self._image_shard_paths.items():
            self._image_fds[name] = os.open(path, os.O_RDONLY)
        self._slat_fd = os.open(self.slat_tar_path, os.O_RDONLY)

    def _build_index(self):
        """Scan image tars and slat tar to build an offset-based index."""
        import time
        t0 = time.time()

        # 1. Scan image shards
        image_index: Dict[str, Dict[str, Any]] = {}
        shard_names = sorted(p for p in os.listdir(self.image_tar_dir) if p.endswith('.tar'))
        self._image_shard_paths = {p: os.path.join(self.image_tar_dir, p) for p in shard_names}

        for shard_name, shard_path in self._image_shard_paths.items():
            for name, offset, size in _scan_tar_fast(shard_path):
                parts = name.split('/')
                if len(parts) != 2:
                    continue
                sample_id, filename = parts
                base, ext = os.path.splitext(filename)
                view_id = '_'.join(base.split('_')[:2])
                if sample_id not in image_index:
                    image_index[sample_id] = {'_shard': shard_name, 'views': {}}
                if view_id not in image_index[sample_id]['views']:
                    image_index[sample_id]['views'][view_id] = {}
                image_index[sample_id]['views'][view_id][ext] = {'offset': offset, 'size': size}

        # 2. Scan slat tar
        slat_index: Dict[str, Dict[str, int]] = {}
        for name, offset, size in _scan_tar_fast(self.slat_tar_path):
            if not name.endswith('.npz'):
                continue
            sample_id = os.path.basename(name).replace('.npz', '')
            slat_index[sample_id] = {'offset': offset, 'size': size}

        # 3. Intersection + voxel counts
        common_ids = sorted(set(image_index.keys()) & set(slat_index.keys()))
        self.samples = []

        # Use slat tar fd for fast voxel count extraction
        slat_fd = os.open(self.slat_tar_path, os.O_RDONLY)
        try:
            for sid in common_ids:
                slat_info = slat_index[sid]
                num_voxels = _get_coords_count_from_npz(slat_fd, slat_info['offset'], slat_info['size'])
                self.samples.append({
                    'sample_id': sid,
                    'image_shard': image_index[sid]['_shard'],
                    'views': image_index[sid]['views'],
                    'slat': slat_info,
                    'num_voxels': num_voxels,
                })
        finally:
            os.close(slat_fd)

        t1 = time.time()
        print(f'[SLatI2ETar] Index built in {t1 - t0:.2f}s, samples: {len(self.samples)}')

    def _save_index(self, path: str):
        with open(path, 'w') as f:
            json.dump({
                'image_shard_paths': self._image_shard_paths,
                'samples': self.samples,
            }, f)

    def _load_index(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self._image_shard_paths = data['image_shard_paths']
        self.samples = data['samples']

    def __len__(self) -> int:
        return len(self.samples)

    def _read_image(self, shard_name: str, offset: int, size: int) -> Image.Image:
        fd = self._image_fds[shard_name]
        os.lseek(fd, offset, os.SEEK_SET)
        raw = os.read(fd, size)
        return Image.open(io.BytesIO(raw))

    def _read_slat(self, offset: int, size: int) -> Dict[str, np.ndarray]:
        fd = self._slat_fd
        os.lseek(fd, offset, os.SEEK_SET)
        raw = os.read(fd, size)
        npz = np.load(io.BytesIO(raw))
        return {k: npz[k] for k in npz.files}

    def process_image(self, image: Image.Image) -> torch.Tensor:
        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_size_ratio = 1.2
        aug_hsize = hsize * aug_size_ratio
        aug_center_offset = [0, 0]
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
        aug_bbox = [
            int(aug_center[0] - aug_hsize),
            int(aug_center[1] - aug_hsize),
            int(aug_center[0] + aug_hsize),
            int(aug_center[1] + aug_hsize),
        ]
        image = image.crop(aug_bbox)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        return image

    def get_instance(self, sample: dict) -> dict:
        # 1. Read slat
        slat_data = self._read_slat(sample['slat']['offset'], sample['slat']['size'])
        coords = torch.tensor(slat_data['coords']).int()
        feats = torch.tensor(slat_data['feats']).float()

        # 2. Pick a random view
        views = list(sample['views'].keys())
        view_id = views[torch.randint(0, len(views), (1,)).item()]
        view_files = sample['views'][view_id]

        # 3. Read image
        img_info = None
        for ext in ('.rgba', '.png', '.jpg', '.jpeg', '.webp'):
            if ext in view_files:
                img_info = view_files[ext]
                break
        if img_info is None:
            raise RuntimeError(f"No image file found for view {view_id}, files={list(view_files.keys())}")

        image = self._read_image(
            sample['image_shard'],
            img_info['offset'],
            img_info['size'],
        )
        relight_image = self.process_image(image)

        if self.normalization is not None:
            feats = (feats - self.mean) / self.std

        return {
            'coords': coords,
            'feats': feats,
            'cond': relight_image,
        }

    def __getitem__(self, index: int) -> dict:
        max_tries = 10
        for i in range(max_tries):
            current_index = (index + i) % len(self)
            try:
                return self.get_instance(self.samples[current_index])
            except Exception as e:
                print(e)
                continue
        raise RuntimeError("Failed to get item after 10 tries")

    def collate_fn(self, batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['coords'].shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            if len(group) == 0:
                continue
            sub_batch = [batch[i] for i in group]
            pack = {}
            coords = []
            feats = []
            layout = []
            start = 0
            for i, b in enumerate(sub_batch):
                coords.append(torch.cat([torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32), b['coords']], dim=-1))
                feats.append(b['feats'])
                layout.append(slice(start, start + b['coords'].shape[0]))
                start += b['coords'].shape[0]
            coords = torch.cat(coords)
            feats = torch.cat(feats)
            pack['x_0'] = SparseTensor(
                coords=coords,
                feats=feats,
            )
            pack['x_0']._shape = torch.Size([len(group), *sub_batch[0]['feats'].shape[1:]])
            pack['x_0'].register_spatial_cache('layout', layout)

            keys = [k for k in sub_batch[0].keys() if k not in ['coords', 'feats']]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]

            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs
