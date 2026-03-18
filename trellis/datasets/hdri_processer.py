import os
# from httpx import get
import numpy as np
from PIL import Image
import cv2

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

import imageio

from simple_ocio import ToneMapper

Tone_mapper = ToneMapper()
Tone_mapper.view = 'AgX'

class HDRI_Preprocessor:
    def __init__(self, envmap_h, envmap_w):
        self.envmap_h = envmap_h
        self.envmap_w = envmap_w
        self.log_scale = 1000

    def load_hdri(self, hdri_file_path):
        # Load HDRI image
        self.hdri = torch.from_numpy(pyexr.read(hdri_file_path)[..., :3]) # [H, W, 3]
    
    def rgb2srgb(self, rgb):
        return torch.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb**(1/2.4) - 0.055)

    def reinhard(self, x, max_point=16):
        # lumi = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
        # lumi = lumi[..., None]
        # y_rein = x * (1 + lumi / (max_point ** 2)) / (1 + lumi)
        # y_rein = x / (1+x)
        y_rein = x * (1 + x / (max_point ** 2)) / (1 + x)
        return y_rein
    
    def perceptual_encoding(self, hdr, L_white=100.0):
        # Photographic tone mapping
        L_w = 0.2126 * hdr[..., 0] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 2]
        L_w_avg = torch.exp(torch.mean(torch.log(L_w + 1e-8)))
        
        # adaptive parameter
        alpha = 0.18 / L_w_avg
        L_d = alpha * L_w
        
        # compress high brightness
        L_d_compress = L_d * (1 + L_d / (L_white**2)) / (1 + L_d)
        
        # apply to RGB channel
        scale = (L_d_compress / (L_w + 1e-8))[..., None]
        result = hdr * scale
        
        return self.rgb2srgb(result.clamp(0, 1))
    
    def hdr_mapping(self, env_hdr, log_scale):
        # map HDR environment maps to LDR and logarithmic representations
        env_ev0 = self.rgb2srgb(self.reinhard(env_hdr, max_point=16).clamp(0, 1))
        # env_log = self.rgb2srgb(torch.log1p(env_hdr) / np.log1p(log_scale)).clamp(0, 1)
        env_log = torch.log1p(10 * env_hdr) / np.log1p(log_scale)
        env_perceptual = self.perceptual_encoding(env_hdr)
        return {
            'env_hdr': env_hdr,    # Original HDR image
            'env_ev0': env_ev0,    # LDR image after tone mapping
            'env_log': env_log,    # Logarithmic scaling
            'env_perceptual': env_perceptual,    # Perceptual encoding
        }

    def get_rotate_hdri_cond(self, hdri_rot_roll):
        # Rotate HDRI image
        '''
        hdri_rot: [3] (0, 0, roll)
        '''
        envir_map_ldr, envir_map_hdr, env_perceptual, envir_map_hdr_raw, view_dirs_world = self.preprcess_envir_map(self.hdri, np.array([0.,0.,hdri_rot_roll]))
        hdri_cond = torch.cat([envir_map_ldr, envir_map_hdr, view_dirs_world], dim=0).float() # [9, H, W]
        return hdri_cond.unsqueeze(0) # [1, 9, H, W]
    
    def rotate_hdri_image(self, envir_map, roll_angle):
        """
        Directly rotate HDRI image (implemented by pixel translation for circular rotation)
        
        Args:
            envir_map: HDRI data, shape is [H, W, 3]
            roll_angle: rotation angle (radians), positive value represents counter-clockwise rotation
        
        Returns:
            rotated_envir_map: rotated HDRI image, shape is [H, W, 3]
        """
        if isinstance(envir_map, np.ndarray):
            envir_map = torch.from_numpy(envir_map)
        
        # calculate the number of pixels to be shifted (convert radians to the ratio of image width)
        # roll_angle range is [-pi, pi], corresponding to the width of the image [-W/2, W/2]
        shift_pixels = int((roll_angle / (2 * np.pi)) * envir_map.shape[1])
        
        # use torch.roll for circular shift (along the width direction)
        rotated_envir_map = torch.roll(envir_map, shifts=shift_pixels, dims=1)
        
        return rotated_envir_map
    
    def fast_process_rotated_hdri(self, envir_map, target_size=(512, 512)):
        """
        Fast process rotated HDRI (skip the get_light sampling step)
        
        Args:
            envir_map: rotated HDRI data, shape is [H, W, 3]
            target_size: target size (H, W)
        
        Returns:
            Tuple of processed tensors
        """
        if isinstance(envir_map, np.ndarray):
            envir_map = torch.from_numpy(envir_map)
        
        # directly perform HDR mapping, skip get_light sampling
        env_ev0 = self.rgb2srgb(self.reinhard(envir_map, max_point=16).clamp(0, 1))

        # resize to target size
        envir_map_ldr = F.interpolate(
            env_ev0.permute(2, 0, 1).unsqueeze(0), 
            size=target_size, mode='nearest', align_corners=None
        ).squeeze()
        
        return envir_map_ldr
    
    def rotate_hdri_and_get_cond(self, envir_map, hdri_rot, tone_mapper=None):
        # ensure envir_map is a tensor
        if isinstance(envir_map, np.ndarray):
            envir_map = torch.from_numpy(envir_map)
        
        # process rotation angle parameter
        if isinstance(hdri_rot, (int, float)):
            roll_angle = float(hdri_rot)
        else:
            hdri_rot = np.array(hdri_rot)
            # only take roll angle (third component)
            roll_angle = float(hdri_rot[2] if len(hdri_rot) > 2 else hdri_rot[0])
        
        rotated_envir_map = self.rotate_hdri_image(envir_map, roll_angle).numpy()
        
        # envir_map_ldr = self.fast_process_rotated_hdri(rotated_envir_map, target_size=(256, 512))
        if tone_mapper is not None:
            envir_map_arr = tone_mapper.hdr_to_ldr(rotated_envir_map)
        else:
            envir_map_arr = Tone_mapper.hdr_to_ldr(rotated_envir_map)
        
        return envir_map_arr  # [H, W, 3]


    def generate_envir_map_dir(self, hdri_rot):
        lat_step_size = np.pi / self.envmap_h
        lng_step_size = 2 * np.pi / self.envmap_w
        theta, phi = torch.meshgrid([torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, self.envmap_h), 
                                    torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, self.envmap_w)], indexing='ij')

        sin_theta = torch.sin(torch.pi / 2 - theta)  # [envH, envW]
        light_area_weight = 4 * torch.pi * sin_theta / torch.sum(sin_theta)  # [envH, envW]
        assert 0 not in light_area_weight, "There shouldn't be light pixel that doesn't contribute"
        light_area_weight = light_area_weight.to(torch.float32).reshape(-1) # [envH * envW, ]

        # phi = phi + np.pi/2 - np.pi - hdri_rot[2]
        phi = phi - hdri_rot[2] - np.pi/2.
        # phi = phi - np.pi - hdri_rot[2]

        view_dirs = torch.stack([   torch.cos(phi) * torch.cos(theta), 
                                    torch.sin(phi) * torch.cos(theta), 
                                    torch.sin(theta)], dim=-1).view(-1, 3)    # [envH * envW, 3]
        light_area_weight = light_area_weight.reshape(self.envmap_h, self.envmap_w)

        return light_area_weight, view_dirs
            
    def get_light(self, hdr_rgb, incident_dir, flip=False, hdr_weight=None, if_weighted=False):
        # flip the image
        envir_map = hdr_rgb.flip(1) if flip else hdr_rgb

        envir_map = envir_map.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
        if hdr_weight is not None:
            hdr_weight = self.light_area_weight.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
        incident_dir = incident_dir.clamp(-1, 1)
        theta = torch.arccos(incident_dir[:, 2]).reshape(-1) # top to bottom: 0 to pi
        phi = torch.atan2(incident_dir[:, 1], incident_dir[:, 0]).reshape(-1) # left to right: pi to -pi
        #  x = -1, y = -1 is the left-top pixel of F.grid_sample's input
        query_y = (theta / np.pi) * 2 - 1 # top to bottom: -1-> 1
        query_y = query_y.clamp(-1+10e-8, 1-10e-8)
        query_x = -phi / np.pi # left to right: -1 -> 1
        query_x = query_x.clamp(-1+10e-8, 1-10e-8)

        grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0).float() # [1, 1, N, 2]

        if if_weighted is False or hdr_weight is None:
            light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)
        else:
            weighted_envir_map = envir_map * hdr_weight
            light_rgbs = F.grid_sample(weighted_envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)

            light_rgbs = light_rgbs / hdr_weight.reshape(-1, 1)
                
        return light_rgbs

    def rotate_and_preprcess_envir_map(self, envir_map, c2w, hdri_rot, flip=False, debug=False):
        self.light_area_weight, self.view_dirs = self.generate_envir_map_dir(hdri_rot)

        env_h, env_w = envir_map.shape[0], envir_map.shape[1]
        axis_aligned_transform = torch.from_numpy(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])).float() # Blender's convention

        R_world2cam = c2w[:3, :3].T

        R_final = axis_aligned_transform @ R_world2cam

        view_dirs_world = self.view_dirs @ R_final # [envH * envW, 3]

        rotated_hdr_rgb = self.get_light(envir_map, view_dirs_world, flip=flip)
        rotated_hdr_rgb = rotated_hdr_rgb.reshape(env_h, env_w, 3)

        # hdr_raw
        mapping_results = self.hdr_mapping(rotated_hdr_rgb, self.log_scale)

        view_dirs_world = view_dirs_world.reshape(env_h, env_w, 3)

        if debug:
            return mapping_results["env_ev0"].permute(2, 0, 1), mapping_results["env_log"].permute(2, 0, 1), mapping_results["env_hdr"].permute(2, 0, 1), mapping_results["env_perceptual"].permute(2, 0, 1), view_dirs_world.permute(2, 0, 1) * 0.5 + 0.5
        
        # resize to 256x256
        envir_map_ldr = F.interpolate(mapping_results["env_ev0"].permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=True).squeeze()
        envir_map_hdr = F.interpolate(mapping_results["env_log"].permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=True).squeeze()
        envir_map_hdr_raw = F.interpolate(mapping_results["env_hdr"].permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=True).squeeze()
        envir_map_perceptual = F.interpolate(mapping_results["env_perceptual"].permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=True).squeeze()
        view_dirs_world = F.interpolate(view_dirs_world.permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=True).squeeze()
        return envir_map_ldr * 2. - 1., envir_map_hdr * 2. - 1., envir_map_hdr_raw, envir_map_perceptual, view_dirs_world

    def preprcess_envir_map(self, envir_map, hdri_rot, flip=False, debug=False):
        self.light_area_weight, self.view_dirs = self.generate_envir_map_dir(hdri_rot)

        env_h, env_w = envir_map.shape[0], envir_map.shape[1]
        axis_aligned_transform = torch.from_numpy(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])).float() 

        view_dirs_world = self.view_dirs @ axis_aligned_transform # [envH * envW, 3]

        rotated_hdr_rgb = self.get_light(envir_map, view_dirs_world, flip=flip)
        rotated_hdr_rgb = rotated_hdr_rgb.reshape(env_h, env_w, 3)

        mapping_results = self.hdr_mapping(rotated_hdr_rgb, self.log_scale)

        view_dirs_world = view_dirs_world.reshape(env_h, env_w, 3)

        if debug:
            return mapping_results["env_ev0"].permute(2, 0, 1), mapping_results["env_log"].permute(2, 0, 1), mapping_results["env_hdr"].permute(2, 0, 1), mapping_results["env_perceptual"].permute(2, 0, 1), view_dirs_world.permute(2, 0, 1) * 0.5 + 0.5
        
        # resize to 256x256
        envir_map_ldr = F.interpolate(mapping_results["env_ev0"].permute(2, 0, 1).unsqueeze(0), size=(512, 512), mode='nearest', align_corners=None).squeeze()
        envir_map_hdr = F.interpolate(mapping_results["env_log"].permute(2, 0, 1).unsqueeze(0), size=(512, 512), mode='nearest', align_corners=None).squeeze()
        envir_map_hdr_raw = F.interpolate(mapping_results["env_hdr"].permute(2, 0, 1).unsqueeze(0), size=(512, 512), mode='nearest', align_corners=None).squeeze()
        envir_map_perceptual = F.interpolate(mapping_results["env_perceptual"].permute(2, 0, 1).unsqueeze(0), size=(512, 512), mode='nearest', align_corners=None).squeeze()
        view_dirs_world = F.interpolate(view_dirs_world.permute(2, 0, 1).unsqueeze(0), size=(512, 512), mode='nearest', align_corners=None).squeeze()
        # return envir_map_ldr * 2. - 1., envir_map_hdr * 2. - 1., envir_map_perceptual * 2. - 1., envir_map_hdr_raw, view_dirs_world
        return envir_map_ldr * 2. - 1., envir_map_hdr - 1., envir_map_perceptual, envir_map_hdr_raw, view_dirs_world
    

if __name__ == "__main__":
    import time
    
    # configure parameters
    exr_file_path = "../../assets/hdris/0012_hdri_4k_hdriskies_4k.exr"
    output_video_path = "../../assets/hdris/0012_hdri_4k_hdriskies_4k_rotation_ldr.mp4"
    
    # use GPU acceleration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # initialize processor
    hdri_processer = HDRI_Preprocessor(envmap_h=512, envmap_w=1024)
    
    # load HDRI
    print(f"Loading HDRI: {exr_file_path}")
    hdri = pyexr.read(exr_file_path)[..., :3]
    hdri = torch.from_numpy(hdri).to(device)
    
    # generate rotation sequence
    num_frames = 60
    rotation_angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
    
    ldr_frames = []
    hdr_frames = []
    perceptual_frames = []
    
    print(f"Generating {num_frames} frames rotation video (using fast mode)...")
    start_time = time.time()
    
    for i, angle in enumerate(rotation_angles):
        if i % 10 == 0:
            print(f"Processing frame {i+1}/{num_frames}, rotation angle: {np.degrees(angle):.1f}°")
        
        # get rotated HDRI condition (using fast mode)
        envir_map_ldr = hdri_processer.rotate_hdri_and_get_cond(
            hdri, 
            hdri_rot=angle,
        )
        
        # get LDR image [3, H, W] -> [H, W, 3]
        ldr_image = envir_map_ldr.cpu().permute(1, 2, 0).numpy()
        ldr_image = np.clip(ldr_image, 0, 1)
        # convert to uint8
        ldr_image = (ldr_image * 255).astype(np.uint8)
        ldr_frames.append(ldr_image)
    
    elapsed_time = time.time() - start_time
    print(f"\nProcessing completed! Total time: {elapsed_time:.2f} seconds ({num_frames/elapsed_time:.2f} frames/second)")
    
    # save video
    print(f"\nSaving LDR video to: {output_video_path}")
    imageio.mimsave(output_video_path, ldr_frames, fps=30)
    print(f"  - LDR: {output_video_path}")
