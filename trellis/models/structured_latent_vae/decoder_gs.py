from typing import *
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from ...modules import sparse as sp
from ...utils.random_utils import hammersley_sequence
from .base import SparseTransformerRegisterSelfBase
from ...representations import Gaussian_view as Gaussian
from ..sparse_elastic_mixin import SparseTransformerElasticMixin
from .hdri_encoder import Hdri_Encoder
from .nerf_encoding import NeRFEncoding

class SLatGaussianDecoder(SparseTransformerRegisterSelfBase):
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
        pe_mode: Literal["ape", "rope"] = "ape",
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

        self.num_register_tokens = num_register_tokens
        self.reg_tokens = nn.Parameter(torch.randn(1, num_register_tokens, model_channels))

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

    def forward(self, x: sp.SparseTensor) -> Tuple[sp.SparseTensor, torch.Tensor]:

        reg_feats = self.reg_tokens.expand(x.shape[0], -1, -1)

        h, reg = super().forward(x, reg=reg_feats)
        h = h.type(x.dtype)
        reg = reg.type(x.dtype)
        return h, reg

        
class ElasticSLatGaussianDecoder(SparseTransformerElasticMixin, SLatGaussianDecoder):
    """
    Slat VAE Gaussian decoder with elastic memory management.
    Used for training with low VRAM.
    """
    pass
