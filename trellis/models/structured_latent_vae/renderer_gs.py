from typing import *
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from ...modules import sparse as sp
from ...utils.random_utils import hammersley_sequence
from .base import SparseTransformerSelfCrossViewRegisterBase, SparseTransformerCrossViewRegisterBase
from ...representations import Gaussian_view as Gaussian
from ..sparse_elastic_mixin import SparseTransformerElasticMixin
from .hdri_encoder import Hdri_Encoder
from .nerf_encoding import NeRFEncoding


class SLatGaussianRenderer(SparseTransformerCrossViewRegisterBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        cond_channels: int,
        num_blocks: int,
        num_register_tokens: int = 16,
        pretrained_decoder_path: str = None,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope", "null"] = "null",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        representation_config: dict = None,
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            cond_channels=cond_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
            qk_rms_norm_cross=qk_rms_norm_cross,
        )
        self.resolution = resolution
        self.rep_config = representation_config
        self._calc_layout()
        self._view_calc_layout()

        self.ob_views_pe = NeRFEncoding(in_dim=3, num_frequencies=12, include_input=True)
        self.ob_views_encoder = nn.Linear(self.ob_views_pe.get_out_dim(), model_channels)
        self.ob_views_norm = nn.RMSNorm(model_channels)

        self.ob_dists_pe = NeRFEncoding(in_dim=1, num_frequencies=12, include_input=True)
        self.ob_dists_encoder = nn.Linear(self.ob_dists_pe.get_out_dim(), model_channels)
        self.ob_dists_norm = nn.RMSNorm(model_channels)

        self.gs_out_layer = nn.Sequential(
            sp.SparseLinear(model_channels, model_channels),
            sp.SparseGELU(),
            sp.SparseLinear(model_channels, self.out_channels)
        )
        self.pbr_out_layer = nn.Sequential(
            sp.SparseLinear(model_channels, model_channels),
            sp.SparseGELU(),
            sp.SparseLinear(model_channels, self.pbr_out_channels)
        )

        self.scale_out_layer = sp.SparseLinear(model_channels, self.view_out_channels)
        
        self.rgb_out_layer = sp.SparseLinear(model_channels, self.hdri_out_channels)
        
        self._build_perturbation()

        self.initialize_weights()

        if pretrained_decoder_path is not None:
            if pretrained_decoder_path.endswith('.safetensors'):
                decoder_weights = load_file(pretrained_decoder_path)
                model_state_dict = self.state_dict()
                for k, v in decoder_weights.items():
                    if k not in model_state_dict:
                        continue
                    elif k in ["input_layer.weight"]:
                        model_state_dict[k][:,:8] = v
                        model_state_dict[k][:,8:16] = v
                        model_state_dict[k][:,16:24] = v
                    else:
                        model_state_dict[k] = v
                
                self.load_state_dict(model_state_dict, strict=True)
            else:
                decoder_weights = torch.load(pretrained_decoder_path, map_location='cpu', weights_only=True)
                model_state_dict = self.state_dict()
                for k, v in decoder_weights.items():
                    # if k not in model_state_dict:
                        # continue
                    if k in ["input_layer.weight"]:
                        model_state_dict[k][:,:8] = v
                        model_state_dict[k][:,8:16] = v
                        model_state_dict[k][:,16:24] = v
                    else:
                        model_state_dict[k] = v
            
                self.load_state_dict(model_state_dict, strict=True)

        print(f"Loaded pretrained decoder from {pretrained_decoder_path}")

        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        super().initialize_weights()

    def _build_perturbation(self) -> None:
        perturbation = [hammersley_sequence(3, i, self.rep_config['num_gaussians']) for i in range(self.rep_config['num_gaussians'])]
        perturbation = torch.tensor(perturbation).float() * 2 - 1
        perturbation = perturbation / self.rep_config['voxel_size']
        perturbation = torch.atanh(perturbation).to(self.device)
        self.register_buffer('offset_perturbation', perturbation)

    def _calc_layout(self) -> None:
        self.layout = {
            '_xyz' : {'shape': (self.rep_config['num_gaussians'], 3), 'size': self.rep_config['num_gaussians'] * 3},
            '_base_color' : {'shape': (self.rep_config['num_gaussians'], 1, 3), 'size': self.rep_config['num_gaussians'] * 3},
            '_scaling' : {'shape': (self.rep_config['num_gaussians'], 3), 'size': self.rep_config['num_gaussians'] * 3},
            '_rotation' : {'shape': (self.rep_config['num_gaussians'], 4), 'size': self.rep_config['num_gaussians'] * 4},
            '_opacity' : {'shape': (self.rep_config['num_gaussians'], 1), 'size': self.rep_config['num_gaussians']}
        }
        start = 0
        for k, v in self.layout.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.out_channels = start

        self.pbr_layout = {
            '_roughness' : {'shape': (self.rep_config['num_gaussians'], 1), 'size': self.rep_config['num_gaussians']},
            '_metallic' : {'shape': (self.rep_config['num_gaussians'], 1), 'size': self.rep_config['num_gaussians']},
            '_pbr1': {'shape': (self.rep_config['num_gaussians'], 1), 'size': self.rep_config['num_gaussians']},
        }
        start = 0
        for k, v in self.pbr_layout.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.pbr_out_channels = start

    def _view_calc_layout(self) -> None:
        self.view_layout = {
            # '_xyz' : {'shape': (self.rep_config['num_gaussians'], 3), 'size': self.rep_config['num_gaussians'] * 3},
            '_scaling_view' : {'shape': (self.rep_config['num_gaussians'], 3), 'size': self.rep_config['num_gaussians'] * 3},
            # '_rotation' : {'shape': (self.rep_config['num_gaussians'], 4), 'size': self.rep_config['num_gaussians'] * 4},
            # '_opacity' : {'shape': (self.rep_config['num_gaussians'], 1), 'size': self.rep_config['num_gaussians']}
        }
        start = 0
        for k, v in self.view_layout.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.view_out_channels = start

        self.hdri_layout = {
            '_rgb' : {'shape': (self.rep_config['num_gaussians'], 32), 'size': self.rep_config['num_gaussians'] * 32},
            '_shadow': {'shape': (self.rep_config['num_gaussians'], 1), 'size': self.rep_config['num_gaussians']},
            '_hdri1': {'shape': (self.rep_config['num_gaussians'], 1), 'size': self.rep_config['num_gaussians']},
            '_hdri2': {'shape': (self.rep_config['num_gaussians'], 1), 'size': self.rep_config['num_gaussians']},
        }

        start = 0
        for k, v in self.hdri_layout.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.hdri_out_channels = start


    def _calc_view_info(self, xyzs: torch.Tensor, cam_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ob_views = xyzs - cam_pos
        ob_dists = torch.norm(ob_views, dim=-1, keepdim=True)
        ob_views = ob_views / ob_dists
        return ob_views, ob_dists

    def to_representation(self, x: sp.SparseTensor) -> List[Gaussian]:
        """
        Args:
            x: The [N x C] sparse tensor output by the network.

        Returns:
            list of representations with view information
        """
        reps = []
        for i in range(x.shape[0]):
            representation = Gaussian(
                sh_degree=0,
                aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
                mininum_kernel_size = self.rep_config['3d_filter_kernel_size'],
                scaling_bias = self.rep_config['scaling_bias'],
                opacity_bias = self.rep_config['opacity_bias'],
                scaling_activation = self.rep_config['scaling_activation']
            )
            xyz = (x.coords[x.layout[i]][:, 1:].float() + 0.5) / self.resolution

            for k, v in self.layout.items():
                if k == '_xyz':
                    offset = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape'])
                    offset = offset * self.rep_config['lr'][k]
                    if self.rep_config['perturb_offset']:
                        offset = offset + self.offset_perturbation
                    offset = torch.tanh(offset) / self.resolution * 0.5 * self.rep_config['voxel_size']
                    _xyz = xyz.unsqueeze(1) + offset
                    setattr(representation, k, _xyz.flatten(0, 1))
                else:
                    feats = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape']).flatten(0, 1)
                    feats = feats * self.rep_config['lr'][k]
                    setattr(representation, k, feats)

            for k, v in self.pbr_layout.items():
                feats = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape']).flatten(0, 1)
                feats = feats * self.rep_config['lr'][k]
                setattr(representation, k, feats)

            for k, v in self.view_layout.items():
                feats = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape']).flatten(0, 1)
                feats = feats * self.rep_config['lr'][k]
                setattr(representation, k, feats)
            
            for k, v in self.hdri_layout.items():
                feats = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape']).flatten(0, 1)
                feats = feats * self.rep_config['lr'][k]
                setattr(representation, k, feats)
            
            reps.append(representation)

        return reps

    def forward(self, h: sp.SparseTensor, reg_feats: torch.Tensor, hdri_cond: torch.Tensor, extrinsics: torch.Tensor) -> List[Gaussian]:
        cam_positions = torch.inverse(extrinsics)[:, :3, 3]
        ob_views = []
        ob_dists = []
        for i in range(len(cam_positions)):
            xyz = (h.coords[h.layout[i]][:, 1:].float() + 0.5) / self.resolution
            ob_view, ob_dist = self._calc_view_info(xyz, cam_positions[i])
            ob_views.append(ob_view)
            ob_dists.append(ob_dist)

        ob_views = torch.cat(ob_views, dim=0)
        ob_dists = torch.cat(ob_dists, dim=0)

        ob_views_emb = self.ob_views_norm(self.ob_views_encoder(self.ob_views_pe(ob_views)))
        ob_dists_emb = self.ob_dists_norm(self.ob_dists_encoder(self.ob_dists_pe(ob_dists)))

        ob_embs = sp.SparseTensor(
            coords=h.coords,
            feats=ob_views_emb + ob_dists_emb,
        )

        h1 = super().forward(h, ob=ob_embs, reg=reg_feats, hdri=hdri_cond)
        h1 = h1.type(h.dtype)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h1 = h1.replace(F.layer_norm(h1.feats, h1.feats.shape[-1:]))

        h_gs = self.gs_out_layer(h + ob_embs)
        h_pbr = self.pbr_out_layer(h + ob_embs)

        h_scale = self.scale_out_layer(h1)
        h_rgbs = self.rgb_out_layer(h1)

        h_out = sp.sparse_cat([h_gs, h_pbr, h_scale, h_rgbs], dim=1)

        reps = self.to_representation(h_out)
        
        return reps

        
class ElasticSLatGaussianRenderer(SparseTransformerElasticMixin, SLatGaussianRenderer):
    """
    Slat VAE Gaussian decoder with elastic memory management.
    Used for training with low VRAM.
    """
    pass
