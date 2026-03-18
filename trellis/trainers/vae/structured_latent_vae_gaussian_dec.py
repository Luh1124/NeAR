from curses import use_default_colors
from typing import *
import copy
import torch
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
import utils3d.torch
import time

from ..basic import BasicTrainer
from ...representations import Gaussian_view as Gaussian
from ...renderers import GaussianRenderer
from ...modules.sparse import SparseTensor
from ...utils.loss_utils import l1_loss, l2_loss, ssim, lpips, psnr_loss, masked_mean, smooth_l1_loss, gamma_correction, get_reflectance_mask
    #   cook_torrance_mask_weight, gan_hinge_loss, get_img_grad_weight, psnr_loss, cosine_loss_per_pixel, masked_mean
from ...utils.data_utils import recursive_to_device


class SLatVaeGaussianDecoderTrainer(BasicTrainer):
    """
    Trainer for structured latent VAE Gaussian Decoder.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
        
        loss_type (str): Loss type. Can be 'l1', 'l2'
        lambda_ssim (float): SSIM loss weight.
        lambda_lpips (float): LPIPS loss weight.
    """
    
    def __init__(
        self,
        *args,
        loss_type: str = 'l1',
        lambda_ssim: float = 0.2,
        lambda_normal: float = 0.2,
        lambda_lpips: float = 0.2,
        lambda_disc: float = 0.2,
        lambda_dist: float = 0.2,
        lambda_depth: float = 0.2,
        lambda_metallic: float = 0.5,
        lambda_base_color: float = 0.5,
        lambda_roughness: float = 0.5,
        lambda_shadow: float = 0.5,
        lambda_brightness: float =0.2,
        weight_brightness: float = 1.0,
        weight_reflectance: float = 1.0,
        weight_confidence: float = 1.0,
        regularizations: Dict = {},
        spec_start_step: int = 2000001,
        normal_consistent_step: int = 30000,
        depth_consistent_step: int = 30000,
        dist_consistent_step: int = 30000,
        shadow_consistent_step: int = 30000,
        brightness_consistent_step: int = 30000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.lambda_ssim = lambda_ssim
        self.lambda_normal = lambda_normal
        self.lambda_lpips = lambda_lpips
        self.lambda_disc = lambda_disc

        self.regularizations = regularizations

        self.lambda_metallic = lambda_metallic
        self.lambda_base_color = lambda_base_color
        self.lambda_roughness = lambda_roughness
        self.lambda_depth = lambda_depth
        self.lambda_dist = lambda_dist
        self.lambda_shadow = lambda_shadow
        self.lambda_brightness = lambda_brightness
        
        self.weight_brightness = weight_brightness
        self.weight_reflectance = weight_reflectance
        self.weight_confidence = weight_confidence

        self.spec_start_step = spec_start_step
        self.normal_consistent_step = normal_consistent_step
        self.depth_consistent_step = depth_consistent_step
        self.dist_consistent_step = dist_consistent_step
        self.shadow_consistent_step = shadow_consistent_step
        self.brightness_consistent_step = brightness_consistent_step
        
        self._init_renderer()
        
    def _init_renderer(self):
        rendering_options = {"near" : 1,
                "far" : 3,
                "bg_color" : 'random',
                "distributed": False,  # If True, use distributed rendering
            }
        self.renderer = GaussianRenderer(rendering_options)
        self.renderer.pipe.kernel_size = self.models['renderer'].rep_config['2d_filter_kernel_size']
        
    def _render_batch(self, reps: List[Gaussian], extrinsics: torch.Tensor, intrinsics: torch.Tensor, opt: edict) -> dict[str, torch.Tensor]:
        """
        Render a batch of representations.

        Args:
            reps: The dictionary of lists of representations.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
        """
        ret = None
        for i, gaussian in enumerate(reps):
            render_pack = self.renderer.render(gaussian, extrinsics[i], intrinsics[i], opt=opt)
            if ret is None:
                ret = {k: [] for k in list(render_pack.keys()) + ['bg_color']}
            for k, v in render_pack.items():
                ret[k].append(v)
            ret['bg_color'].append(self.renderer.bg_color)
        for k, v in ret.items():
            ret[k] = torch.stack(v, dim=0) 
        return ret

    @torch.no_grad()
    def _get_status(self, z: SparseTensor, reps: List[Gaussian]) -> Dict:
        xyz = torch.cat([g.get_xyz for g in reps], dim=0)
        xyz_base = (z.coords[:, 1:].float() + 0.5) / self.models['renderer'].resolution - 0.5
        offset = xyz - xyz_base.unsqueeze(1).expand(-1, self.models['renderer'].rep_config['num_gaussians'], -1).reshape(-1, 3)
        status = {
            'xyz': xyz,
            'offset': offset,
            'scale': torch.cat([g.get_scaling for g in reps], dim=0),
            'scale_view': torch.cat([g.get_scaling_view for g in reps], dim=0),
            'opacity': torch.cat([g.get_opacity for g in reps], dim=0),
        }

        for k in list(status.keys()):
            status[k] = {
                'mean': status[k].mean().item(),
                'max': status[k].max().item(),
                'min': status[k].min().item(),
            }
            
        return status
    
    def _get_regularization_loss(self, reps: List[Gaussian]) -> Tuple[torch.Tensor, Dict]:
        loss = 0.0
        terms = {}
        if 'lambda_vol' in self.regularizations:
            scales = torch.cat([g.get_scaling for g in reps], dim=0)   # [N x 3]
            volume = torch.prod(scales, dim=1)  # [N]
            scales_view = torch.cat([g.get_scaling_view for g in reps], dim=0)   # [N x 3]
            volume_view = torch.prod(scales_view, dim=1)  # [N]
            terms[f'reg_vol'] = volume.mean() + volume_view.mean()
            loss = loss + self.regularizations['lambda_vol'] * terms[f'reg_vol']
        if 'lambda_opacity' in self.regularizations:
            opacity = torch.cat([g.get_opacity for g in reps], dim=0)
            terms[f'reg_opacity'] = (opacity - 1).pow(2).mean()
            loss = loss + self.regularizations['lambda_opacity'] * terms[f'reg_opacity']
        return loss, terms
    
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        # d_weight = d_weight * self.discriminator_weight
        d_weight = d_weight * 0.5
        return d_weight
    
    def training_losses(
        self,
        latents: SparseTensor,
        # color_latents: SparseTensor,
        # pbr_latents: SparseTensor,
        Roughness: torch.Tensor,
        Metallic: torch.Tensor,
        Basecolor: torch.Tensor,
        shadow: torch.Tensor,
        brightness: torch.Tensor,
        normal: torch.Tensor,
        depth: torch.Tensor,
        hdri: torch.Tensor,
        hdri_cond: torch.Tensor,
        hdri_rot: torch.Tensor,
        image: torch.Tensor,
        alpha: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        train_discriminator: bool = False,  # 新增判别器训练标志  
        return_aux: bool = False,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            latents: The [N x * x C] sparse latents
            hdri_cond: The [N x C] tensor of HDRI conditions.
            image: The [N x 3 x H x W] tensor of images.
            alpha: The [N x H x W] tensor of alpha channels.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
            return_aux: Whether to return auxiliary information.
        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        # use_asg = self.step > self.spec_start_step and False
        reps = self.get_representation(latents=latents, hdri_cond = hdri_cond, extrinsics = extrinsics)
        self.renderer.rendering_options.resolution = image.shape[-1]
                
        opt = edict(
            neural_basis=self.training_models['neural_basis'],
        )
        render_results = self._render_batch(reps, extrinsics, intrinsics, opt)
        
        terms = edict(loss = 0.0, rec = 0.0)
        
        # rec_image = render_results['color'] if use_asg else render_results['color'] + render_results['specular']
        rec_image = render_results['color']
        
        gt_image = image * alpha + (1 - alpha) * render_results['bg_color'][..., None, None]
        gt_normal = normal * alpha
        
        # cook_torrance_mask = cook_torrance_mask_weight(
        #     metallic=Metallic,
        #     roughness=Roughness,
        #     epsilon=1e-6
        # )
        # weight = 1 + cook_torrance_mask * alpha * 2

        weight = 1 + brightness * self.weight_brightness + get_reflectance_mask(Roughness, Metallic) * self.weight_reflectance

        # confidence = render_results['hdri1']
        # weight = weight + confidence * self.weight_confidence

        if self.loss_type == 'l1':
            log_rec_image = torch.log(rec_image + 1.)
            log_gt_image = torch.log(gt_image + 1.)
            terms["l1"] = l1_loss(log_rec_image, log_gt_image, weight=weight)
            terms["rec"] = terms["rec"] + terms["l1"]
        elif self.loss_type == 'l2':
            terms["l2"] = l2_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l2"]
        elif self.loss_type == 'smooth_l1':
            terms["l1"] = l1_loss(rec_image, gt_image).detach()
            smooth_l1 = torch.nn.functional.smooth_l1_loss(rec_image, gt_image, reduction='none')  
            # weighted_smooth_l1 = smooth_l1 * weight
            weighted_smooth_l1 = smooth_l1
            terms["smooth_l1"] = weighted_smooth_l1.mean()  
            terms["rec"] = terms["rec"] + terms["smooth_l1"]
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        
        if self.lambda_normal > 0 and self.step > self.normal_consistent_step and False:
            rendered_normal = render_results["normal"] * alpha
            depth_normal = render_results["normal_from_depth"] * alpha
            depth_normal_view = render_results["normal_from_depth_view"] * alpha
            
            normal_error1 = (1 - (gt_normal * depth_normal).sum(dim=0))[None]
            normal_error2 = (1 - (gt_normal * rendered_normal).sum(dim=0))[None]
            normal_error3 = (1 - (gt_normal * depth_normal_view).sum(dim=0))[None]
            terms["normal"] = normal_error1.mean() + normal_error2.mean() + normal_error3.mean()
            terms["rec"] = terms["rec"] + self.lambda_normal * terms["normal"]
        
        if self.lambda_dist > 0 and self.step > self.dist_consistent_step and False:
            terms["dist"] = render_results["distort"].mean()
            terms["dist"] = terms["dist"] * self.lambda_dist
            terms["rec"] = terms["rec"] + terms["dist"]
        
        if self.lambda_depth > 0 and self.step > self.depth_consistent_step and False:
            gt_depth = torch.where(depth > 0.0, depth - 1.5, torch.zeros_like(depth)) 
            rec_depth = torch.where(render_results['depth'] > 0.0, render_results['depth'] - 1.5, torch.zeros_like(render_results['depth']))
            rec_depth_view = torch.where(render_results['depth_view'] > 0.0, render_results['depth_view'] - 1.5, torch.zeros_like(render_results['depth_view']))
            terms["depth"] = l1_loss(rec_depth, gt_depth) + l1_loss(rec_depth_view, gt_depth)
            terms["depth"] = terms["depth"] * self.lambda_depth
            terms["rec"] = terms["rec"] + terms["depth"]

        if self.lambda_metallic > 0:
            gt_metallic = Metallic * alpha 
            rec_metallic = render_results['metallic'] * alpha
            terms["metallic"] = l1_loss(rec_metallic, gt_metallic)
            terms["rec"] = terms["rec"] + self.lambda_metallic * terms["metallic"]
        
        if self.lambda_base_color > 0:
            gt_base_color = Basecolor * alpha + (1 - alpha) * render_results['bg_color'][..., None, None]
            rec_base_color = render_results['base_color'] * alpha + (1 - alpha) * render_results['bg_color'][..., None, None]
            terms["base_color"] = l1_loss(rec_base_color, gt_base_color)
            terms["rec"] = terms["rec"] + self.lambda_base_color * terms["base_color"]
        
        if self.lambda_roughness > 0:
            gt_roughness = Roughness * alpha
            rec_roughness = render_results['roughness'] * alpha
            terms["roughness"] = l1_loss(rec_roughness, gt_roughness)
            # terms["roughness"] = masked_mean((render_results['roughness'] - Roughness).abs(), alpha)
            terms["rec"] = terms["rec"] + self.lambda_roughness * terms["roughness"]

        if self.lambda_shadow > 0 and self.step > self.shadow_consistent_step:
            gt_shadow = shadow * alpha
            rec_shadow = render_results['shadow'] * alpha
            terms["shadow"] = l1_loss(rec_shadow, gt_shadow)
            terms["rec"] = terms["rec"] + self.lambda_shadow * terms["shadow"]

        if self.lambda_brightness > 0 and self.step > self.brightness_consistent_step and False:
            gt_brightness = brightness
            rec_brightness = render_results['brightness'] * alpha
            terms["brightness"] = l1_loss(rec_brightness, gt_brightness)
            terms["rec"] = terms["rec"] + self.lambda_brightness * terms["brightness"]

        gamma_correction_rec_image = gamma_correction(rec_image, gamma=2.2)
        gamma_correction_gt_image = gamma_correction(gt_image, gamma=2.2)

        if self.lambda_ssim > 0:
            terms["ssim"] = 1 - ssim(gamma_correction_rec_image, gamma_correction_gt_image)
            terms["rec"] = terms["rec"] + self.lambda_ssim * terms["ssim"]
        if self.lambda_lpips > 0:
            terms["lpips"] = lpips(gamma_correction_rec_image, gamma_correction_gt_image)
            terms["rec"] = terms["rec"] + self.lambda_lpips * terms["lpips"]

        terms["psnr"] = psnr_loss(gamma_correction_rec_image, gamma_correction_gt_image)
        
        terms["loss"] = terms["rec"] + terms["loss"]

        reg_loss, reg_terms = self._get_regularization_loss(reps)
        terms.update(reg_terms)
        terms["loss"] = terms["loss"] + reg_loss

        status = self._get_status(latents, reps)

        # if train_discriminator:            
        #     # 判别器训练，loss 仅包含判别器 hinge loss  
        #     pred_real = self.models['discriminator'](gt_image, cond_disc.detach())  # 真实图detach
        #     loss_real = gan_hinge_loss(pred_real, is_real=True)  

        #     pred_fake = self.models['discriminator'](rec_image.detach(), cond_disc.detach())  # 生成图detach
        #     loss_fake = gan_hinge_loss(pred_fake, is_real=False)  

        #     disc_loss = (loss_real + loss_fake) * self.lambda_disc  # 判别器loss
        #     terms["disc_loss"] = disc_loss  
        #     terms["loss"] = disc_loss  # 判别器训练总loss直接赋值  

        # elif self.step > self.disc_start_step:
        #     # 生成器训练阶段，期望生成图被判别器判为真  
        #     pred_fake_for_g = self.models['discriminator'](rec_image, cond_disc)  
        #     gan_loss = gan_hinge_loss(pred_fake_for_g, is_real=True)  # 生成器“骗”判别器  

        #     terms["gan_loss"] = gan_loss  
        #     terms["loss"] += gan_loss  
        #     terms["rec"] += gan_loss  
        # else:
        #     pass

        if return_aux:
            return terms, status, {'rec_image': rec_image, 'gt_image': gt_image}
        return terms, status

    def get_representation(self, latents: SparseTensor, hdri_cond: torch.Tensor, extrinsics: torch.Tensor) -> List[Gaussian]:
        hdri_cond = self.training_models['hdri_encoder'](hdri_cond)
        h, reg_feats = self.training_models['decoder'](latents)
        reps = self.training_models['renderer'](h, reg_feats, hdri_cond, extrinsics)
        return reps

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
        mode: str = 'val',
    ) -> Dict:
        dataloader = DataLoader(
            # copy.deepcopy(self.dataset),
            copy.deepcopy(self.val_dataset if mode == 'val' else self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        ret_dict = {}
        gt_images = []
        gt_normals = []
        gt_roughness = []
        gt_metallic = []
        gt_base_color = []
        gt_depth = []
        gt_shadow = []
        gt_brightness = []
        gt_relf = []
        gt_slat_path = []
        exts = []
        ints = []
        reps = []

        data_list = []

        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = recursive_to_device(data, 'cuda')
            args = {k: v[:batch] for k, v in args.items()}
            data_list.append(args)
            
            gt_images.append(gamma_correction(args['image'] * args['alpha']))
            gt_normals.append(args['normal'] * args['alpha'])
            gt_roughness.append(args['Roughness'] * args['alpha'])
            gt_metallic.append(args['Metallic'] * args['alpha'])
            gt_base_color.append(args['Basecolor'] * args['alpha'])
            gt_depth.append(torch.where(args['depth'] > 0.0, args['depth'] - 1.5, torch.zeros_like(args['depth'])))
            gt_shadow.append(args['shadow'] * args['alpha'])
            gt_brightness.append(args['brightness'] * args['alpha'])
            gt_relf.append(get_reflectance_mask(args['Roughness'], args['Metallic']))

            gt_slat_path.extend(args['slat_path'])

            exts.append(args['extrinsics'])
            ints.append(args['intrinsics'])
            rep = self.get_representation(latents=args['latents'], hdri_cond=args['hdri_cond'], extrinsics=args['extrinsics'])
            reps.extend(rep)

        gt_images = torch.cat(gt_images, dim=0)
        ret_dict.update({f'gt_image': {'value': gt_images, 'type': 'image'}})
        ret_dict.update({f'gt_normal': {'value': torch.cat(gt_normals, dim=0) * 0.5 + 0.5, 'type': 'image'}})
        ret_dict.update({f'gt_roughness': {'value': torch.cat(gt_roughness, dim=0), 'type': 'image'}})
        ret_dict.update({f'gt_metallic': {'value': torch.cat(gt_metallic, dim=0), 'type': 'image'}})
        ret_dict.update({f'gt_base_color': {'value': torch.cat(gt_base_color, dim=0), 'type': 'image'}})
        ret_dict.update({f'gt_depth': {'value': torch.cat(gt_depth, dim=0), 'type': 'image'}})
        ret_dict.update({f'gt_shadow': {'value': torch.cat(gt_shadow, dim=0), 'type': 'image'}})
        ret_dict.update({f'gt_brightness': {'value': torch.cat(gt_brightness, dim=0), 'type': 'image'}})
        ret_dict.update({f'gt_relf': {'value': torch.cat(gt_relf, dim=0), 'type': 'image'}})
        ret_dict.update({f'gt_slat_path': {'value': gt_slat_path, 'type': 'string'}})

        # render single view
        exts = torch.cat(exts, dim=0)
        ints = torch.cat(ints, dim=0)
        self.renderer.rendering_options.bg_color = (0, 0, 0)
        self.renderer.rendering_options.resolution = gt_images.shape[-1]

        opt = edict(
            neural_basis=self.training_models['neural_basis'],
        )

        render_results = self._render_batch(reps, exts, ints, opt)

        ret_dict.update({f'rec_image': {'value': gamma_correction(render_results['color']), 'type': 'image'}})
        ret_dict.update({f'rec_normal': {'value': render_results['normal'] * 0.5 + 0.5, 'type': 'image'}})
        ret_dict.update({f'rec_depth_normal': {'value': render_results['normal_from_depth'] * 0.5 + 0.5, 'type': 'image'}})
        ret_dict.update({f'rec_base_color': {'value': render_results['base_color'], 'type': 'image'}})
        ret_dict.update({f'rec_metallic': {'value': render_results['metallic'], 'type': 'image'}})
        ret_dict.update({f'rec_roughness': {'value': render_results['roughness'], 'type': 'image'}})
        ret_dict.update({f'rec_shadow': {'value': render_results['shadow'], 'type': 'image'}})
        ret_dict.update({f'rec_brightness': {'value': render_results['brightness'], 'type': 'image'}})
        ret_dict.update({f'rec_pbr1': {'value': render_results['pbr1'], 'type': 'image'}})
        ret_dict.update({f'rec_hdri1': {'value': render_results['hdri1'], 'type': 'image'}})
        ret_dict.update({f'rec_hdri2': {'value': render_results['hdri2'], 'type': 'image'}})
        ret_dict.update({f'rec_nush1': {'value': render_results['nush1'], 'type': 'image'}})

        render_results['depth'] = torch.where(render_results['depth'] > 0.0, render_results['depth'] - 1.5, torch.zeros_like(render_results['depth'])) # render depth for visualizer
        ret_dict.update({f'rec_depth': {'value': render_results['depth'], 'type': 'image'}})
        
        ret_dict.update({f'rec_normal_view': {'value': render_results['normal_view'] * 0.5 + 0.5, 'type': 'image'}})
        ret_dict.update({f'rec_depth_normal_view': {'value': render_results['normal_from_depth_view'] * 0.5 + 0.5, 'type': 'image'}})
        
        render_results['depth_view'] = torch.where(render_results['depth_view'] > 0.0, render_results['depth_view'] - 1.5, torch.zeros_like(render_results['depth_view'])) # render depth for visualizer
        ret_dict.update({f'rec_depth_view': {'value': render_results['depth_view'], 'type': 'image'}})


        # render multiview
        self.renderer.rendering_options.resolution = 512
        ## Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        ## render each view
        multiview_dict = {
            "multiview_images": [],
            "multiview_base_colors": [],
            "multiview_metallics": [],
            "multiview_roughnesses": [],
            "multiview_render_normals": [],
            "multiview_render_depth_normals": [],
            "multiview_depths": [],
            "multiview_shadow": [],
            "multiview_brightness": [],
            "multiview_normal_view": [],
            "multiview_depth_normal_view": [],
            "multiview_depth_view": [],
            "multiview_pbr1": [],
            "multiview_hdri1": [],
            "multiview_hdri2": [],
            "multiview_nush1": [],
        }

        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            extrinsics = extrinsics.unsqueeze(0).expand(num_samples, -1, -1)
            intrinsics = intrinsics.unsqueeze(0).expand(num_samples, -1, -1)

            reps = []
            start_idx = 0
            for i, args in enumerate(data_list):
                end_idx = start_idx + len(args['hdri'])
                rep = self.get_representation(latents=args['latents'], hdri_cond=args['hdri_cond'], extrinsics=extrinsics[start_idx:end_idx])
                reps.extend(rep)
                start_idx = end_idx

            render_results = self._render_batch(reps, extrinsics, intrinsics, opt)
            render_results['depth'] = torch.where(render_results['depth'] > 0.0, render_results['depth'] - 1.5, torch.zeros_like(render_results['depth'])) # render depth for visualizer
            render_results['depth_view'] = torch.where(render_results['depth_view'] > 0.0, render_results['depth_view'] - 1.5, torch.zeros_like(render_results['depth_view'])) # render depth for visualizer
            # Store results
            multiview_dict["multiview_images"].append(gamma_correction(render_results['color']))
            multiview_dict["multiview_render_normals"].append(render_results['normal'] * 0.5 + 0.5)
            multiview_dict["multiview_render_depth_normals"].append(render_results['normal_from_depth'] * 0.5 + 0.5)
            multiview_dict["multiview_base_colors"].append(render_results['base_color'])
            multiview_dict["multiview_metallics"].append(render_results['metallic'])
            multiview_dict["multiview_roughnesses"].append(render_results['roughness'])
            multiview_dict["multiview_shadow"].append(render_results['shadow'])
            multiview_dict["multiview_depths"].append(render_results['depth'])
            multiview_dict["multiview_brightness"].append(render_results['brightness'])
            multiview_dict["multiview_normal_view"].append(render_results['normal_view'] * 0.5 + 0.5)
            multiview_dict["multiview_depth_normal_view"].append(render_results['normal_from_depth_view'] * 0.5 + 0.5)
            multiview_dict["multiview_depth_view"].append(render_results['depth_view'])
            multiview_dict["multiview_pbr1"].append(render_results['pbr1'])
            multiview_dict["multiview_hdri1"].append(render_results['hdri1'])
            multiview_dict["multiview_hdri2"].append(render_results['hdri2'])
            multiview_dict["multiview_nush1"].append(render_results['nush1'])

        for k, v in multiview_dict.items():
            concatenated = torch.cat([
                torch.cat(v[:2], dim=-2),
                torch.cat(v[2:], dim=-2),
            ], dim=-1)
            ret_dict.update({f'{k}': {'value': concatenated, 'type': 'image'}})

        self.renderer.rendering_options.bg_color = 'random'
                                    
        return ret_dict
