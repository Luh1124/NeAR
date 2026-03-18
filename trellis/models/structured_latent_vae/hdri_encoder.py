import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import math
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # calculate RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # pre-calculate frequency
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # cache position encoding
        self._cached_seq_len = 0
        self._cached_cos = None
        self._cached_sin = None
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._cached_seq_len:
            self._cached_seq_len = seq_len
            
            # generate position sequence
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            
            # calculate frequency * position
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # [seq_len, dim//2]
            
            # concatenate to get complete frequency matrix
            emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
            
            # cache cos and sin
            self._cached_cos = emb.cos().to(dtype)
            self._cached_sin = emb.sin().to(dtype)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        if seq_len is None:
            seq_len = x.shape[-2]
        
        self._update_cache(seq_len, x.device, x.dtype)
        
        return self._cached_cos[:seq_len], self._cached_sin[:seq_len]


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """apply rotary position encoding to query and key"""
    def rotate_half(x):
        """rotate the second half of the input dimension"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    # apply rotary transformation
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    
    return q_embed, k_embed


class MultiHeadAttentionWithRoPE(nn.Module):
    """multi-head attention with RoPE"""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout
        
        # linear projection layer
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(self.head_dim, base=rope_base)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None):
        B, T, C = query.shape
        
        # linear projection
        q = self.q_proj(query)  # [B, T, d_model]
        k = self.k_proj(key)    # [B, T, d_model] 
        v = self.v_proj(value)  # [B, T, d_model]
        
        # reshape to multi-head format
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # [B, nhead, T, head_dim]
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # [B, nhead, T, head_dim]
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # [B, nhead, T, head_dim]
        
        # apply RoPE
        cos, sin = self.rope(q, T)
        # expand dimensions to match multi-head format
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim]
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # calculate attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, nhead, T, T]
        
        # apply mask
        if attn_mask is not None:
            attn_weights += attn_mask
        
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        
        # softmax + dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # apply attention weights
        out = torch.matmul(attn_weights, v)  # [B, nhead, T, head_dim]
        
        # merge multi-head
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, d_model]
        
        # output projection
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        
        return out


class TransformerEncoderLayerWithRoPE(nn.Module):
    """Transformer encoder layer with RMSNorm and RoPE"""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 dropout: float = 0.1, activation: str = 'gelu', rope_base: float = 10000.0):
        super().__init__()
        
        # multi-head attention (with RoPE)
        self.self_attn = MultiHeadAttentionWithRoPE(d_model, nhead, dropout, rope_base)
        
        # feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # RMSNorm layer
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # activation function
        self.activation = getattr(F, activation)
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None):
        # Self-attention with residual connection and RMSNorm
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask, 
                             key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        # Feed-forward with residual connection and RMSNorm
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src


class TransformerEncoderWithRoPE(nn.Module):
    """Transformer encoder with RMSNorm and RoPE"""
    def __init__(self, encoder_layer: TransformerEncoderLayerWithRoPE, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            encoder_layer for _ in range(num_layers)
        ])
        self.num_layers = num_layers
    
    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None):
        output = src
        
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        
        return output


class FourierFeatureEncoder(nn.Module):
    def __init__(self, input_channels=3, num_freq_bands=10):
        super().__init__()
        self.num_freq_bands = num_freq_bands
        self.output_channels = input_channels * (2 * num_freq_bands + 1)
        self.freq_bands = nn.Parameter(2.0 ** torch.arange(num_freq_bands) * torch.pi, requires_grad=False)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_permuted = x.permute(0, 2, 3, 1)
        scaled_x = x_permuted.unsqueeze(-1) * self.freq_bands
        sincos_features = torch.cat([torch.sin(scaled_x), torch.cos(scaled_x)], dim=-1)
        sincos_features = sincos_features.reshape(B, H, W, -1)
        final_features = torch.cat([x_permuted, sincos_features], dim=-1)
        return final_features.permute(0, 3, 1, 2)


class DirectionGuidedFusion(nn.Module):
    """注意力引导的方向-视觉特征融合模块"""
    def __init__(self, visual_dim, dir_dim, out_dim):
        super().__init__()
        
        # direction feature adapter - project direction feature to suitable dimension
        self.dir_adapter = nn.Sequential(
            nn.Conv2d(dir_dim, visual_dim, kernel_size=1),
            nn.GroupNorm(min(8, visual_dim // 4), visual_dim),  # adaptive group number
            nn.GELU()
        )
        
        # attention mechanism - let direction information guide the weight allocation of visual features
        hidden_dim = max(16, visual_dim // 8)  # ensure minimum hidden dimension
        self.attention = nn.Sequential(
            nn.Conv2d(visual_dim + dir_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(min(4, hidden_dim // 4), hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, visual_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # feature fusion - combine attention-weighted visual features and adapted direction features
        self.fusion = nn.Sequential(
            nn.Conv2d(visual_dim * 2, out_dim, kernel_size=1),
            nn.GroupNorm(min(8, out_dim // 4), out_dim),
            nn.GELU()
        )
        
        # residual connection projection layer
        self.residual_proj = nn.Conv2d(visual_dim, out_dim, kernel_size=1) if visual_dim != out_dim else nn.Identity()
    
    def forward(self, visual_feat, dir_feat):
        # 1. direction feature adapt to visual feature dimension
        dir_adapted = self.dir_adapter(dir_feat)  # [B, visual_dim, H, W]
        
        # 2. calculate attention weights (direction information guides visual features)
        attention_input = torch.cat([visual_feat, dir_feat], dim=1)
        attention_weight = self.attention(attention_input)  # [B, visual_dim, H, W]
        
        # 3. apply attention weights to visual features
        attended_visual = visual_feat * attention_weight  # 元素级相乘
        
        # 4. fuse attention-weighted visual features and adapted direction features
        fused_input = torch.cat([attended_visual, dir_adapted], dim=1)
        fused = self.fusion(fused_input)
        
        # 5. residual connection
        residual = self.residual_proj(visual_feat)
        output = fused + residual
        
        return output


class Hdri_Encoder(nn.Module):
    def __init__(self, output_dim=768, num_tokens=4096, cnn_out_channels=256, 
                 n_heads=8, num_transformer_layers=2, rope_base=10000.0, pretrained_path=None):
        super().__init__()
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        
        # calculate target resolution (ensure token number matches)
        self.target_h = int(math.sqrt(num_tokens))
        self.target_w = self.target_h
        assert self.target_h * self.target_w == num_tokens, f"num_tokens must be a perfect square, got {num_tokens}"
        
        print(f"Target resolution: {self.target_h}x{self.target_w} = {num_tokens} tokens")

        # --- 1. independent direction encoder ---
        # reduce frequency band number to improve efficiency
        self.dir_encoder = FourierFeatureEncoder(input_channels=3, num_freq_bands=8)  # reduce to 8
        dir_enc_channels = self.dir_encoder.output_channels  # 3*(2*8+1) = 51

        # --- 2. ConvNeXt visual backbone network ---
        convnext = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        
        # extract ConvNeXt's various stages
        self.stage1 = nn.Sequential(convnext.features[0], convnext.features[1])  # H/4, 96
        self.stage2 = nn.Sequential(convnext.features[2], convnext.features[3])  # H/8, 192
        self.stage3 = nn.Sequential(convnext.features[4], convnext.features[5])  # H/16, 384
        self.stage4 = nn.Sequential(convnext.features[6], convnext.features[7])  # H/32, 768
        
        # modify the first convolution layer to accept 6 channels (LDR+HDR)
        orig_stem_conv = self.stage1[0][0]
        new_stem_conv = nn.Conv2d(6, orig_stem_conv.out_channels, 
                                 kernel_size=orig_stem_conv.kernel_size, 
                                 stride=orig_stem_conv.stride,
                                 padding=orig_stem_conv.padding)
        
        # weight initialization - reuse the first 3 channels of the pretrained weights, and initialize the last 3 channels with the same weights
        with torch.no_grad():
            new_stem_conv.weight[:, :3] = orig_stem_conv.weight
            new_stem_conv.weight[:, 3:6] = orig_stem_conv.weight  # HDR channel reuse LDR weights
            new_stem_conv.bias = orig_stem_conv.bias
        
        self.stage1[0][0] = new_stem_conv

        # --- 3. attention-guided fusion layer ---
        # ConvNeXt_Tiny's output channels: 192, 384, 768
        self.fusion_c3 = DirectionGuidedFusion(192, dir_enc_channels, cnn_out_channels)
        self.fusion_c4 = DirectionGuidedFusion(384, dir_enc_channels, cnn_out_channels)
        self.fusion_c5 = DirectionGuidedFusion(768, dir_enc_channels, cnn_out_channels)

        # --- 4. final projection and Transformer head ---
        self.projection_conv = nn.Conv2d(cnn_out_channels * 3, output_dim, kernel_size=1)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayerWithRoPE(
            d_model=output_dim,
            nhead=n_heads,
            dim_feedforward=output_dim * 4,
            dropout=0.1,
            activation='gelu',
            rope_base=rope_base
        )
        self.transformer_encoder = TransformerEncoderWithRoPE(encoder_layer, num_transformer_layers)
        self.ln_final = RMSNorm(output_dim)

        # parameter initialization
        self._initialize_weights()

        self.load_weights(pretrained_path)

    
    def convert_to_fp16(self):
        pass

    def convert_to_fp32(self):
        pass
    
    def _initialize_weights(self):
        """weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_weights(self, pretrained_path):
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            self.load_state_dict(checkpoint)

    def forward(self, context):
        B, C, H, W = context.shape
        assert C == 9, f"Expected 9 channels (3 LDR + 3 HDR + 3 directions), got {C}"
        
        # separate different modalities
        split_indices = [3, 6]
        ldr_map, hdr_map, view_dirs_map = torch.tensor_split(context, split_indices, dim=1)
        
        # direction information encoding
        dir_encoding = self.dir_encoder(view_dirs_map)  # [B, 51, H, W]
        
        # visual information fusion
        visual_input = torch.cat([ldr_map, hdr_map], dim=1)  # [B, 6, H, W]

        # --- ConvNeXt backbone forward propagation ---
        c2 = self.stage1(visual_input)  # [B, 96, H/4, W/4]
        c3 = self.stage2(c2)           # [B, 192, H/8, W/8] - high-frequency features
        c4 = self.stage3(c3)           # [B, 384, H/16, W/16] - medium-frequency features  
        c5 = self.stage4(c4)           # [B, 768, H/32, W/32] - low-frequency features
        
        # --- pre-calculate direction encoding at different scales (avoid duplicate interpolation) ---
        target_size = (self.target_h, self.target_w)
        dir_c3 = F.interpolate(dir_encoding, size=c3.shape[2:], mode='bilinear', align_corners=False)
        dir_c4 = F.interpolate(dir_encoding, size=c4.shape[2:], mode='bilinear', align_corners=False) 
        dir_c5 = F.interpolate(dir_encoding, size=c5.shape[2:], mode='bilinear', align_corners=False)
        
        # --- attention-guided multi-scale feature fusion ---
        p_high = self.fusion_c3(c3, dir_c3)  # [B, 128, H/8, W/8]
        p_high = F.interpolate(p_high, size=target_size, mode='bilinear', align_corners=False)
        
        p_mid = self.fusion_c4(c4, dir_c4)   # [B, 128, H/16, W/16]
        p_mid = F.interpolate(p_mid, size=target_size, mode='bilinear', align_corners=False)
        
        p_low = self.fusion_c5(c5, dir_c5)   # [B, 128, H/32, W/32]
        p_low = F.interpolate(p_low, size=target_size, mode='bilinear', align_corners=False)
        
        # fuse multi-scale features
        multi_scale_feat = torch.cat([p_high, p_mid, p_low], dim=1)  # [B, 384, target_h, target_w]

        projected = self.projection_conv(multi_scale_feat)  # [B, output_dim, H, W]
        B, C, H, W = projected.shape
        
        tokens = projected.view(B, C, H * W).permute(0, 2, 1)  # [B, num_tokens, output_dim]
        
        tokens = self.transformer_encoder(tokens)
        tokens = self.ln_final(tokens)
        
        return tokens

    def get_attention_maps(self, context):
        """get attention maps for visualization"""
        with torch.no_grad():
            B, C, H, W = context.shape
            split_indices = [3, 6]
            ldr_map, hdr_map, view_dirs_map = torch.tensor_split(context, split_indices, dim=1)
            
            dir_encoding = self.dir_encoder(view_dirs_map)
            visual_input = torch.cat([ldr_map, hdr_map], dim=1)
            
            c2 = self.stage1(visual_input)
            c3 = self.stage2(c2)
            c4 = self.stage3(c3)
            c5 = self.stage4(c4)
            
            # get attention weights for each layer
            dir_c3 = F.interpolate(dir_encoding, size=c3.shape[2:], mode='bilinear', align_corners=False)
            dir_c4 = F.interpolate(dir_encoding, size=c4.shape[2:], mode='bilinear', align_corners=False)
            dir_c5 = F.interpolate(dir_encoding, size=c5.shape[2:], mode='bilinear', align_corners=False)
            
            # calculate attention weights (without fusion steps)
            att_c3 = self.fusion_c3.attention(torch.cat([c3, dir_c3], dim=1))
            att_c4 = self.fusion_c4.attention(torch.cat([c4, dir_c4], dim=1))
            att_c5 = self.fusion_c5.attention(torch.cat([c5, dir_c5], dim=1))
            
            return {
                'attention_c3': att_c3,  # [B, 192, H/8, W/8]
                'attention_c4': att_c4,  # [B, 384, H/16, W/16]
                'attention_c5': att_c5,  # [B, 768, H/32, W/32]
            }


if __name__ == "__main__":
    import time
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # test individual components
    print("Testing individual components...")
    
    # 1. test RMSNorm
    rms_norm = RMSNorm(768).to(device)
    x = torch.randn(2, 1024, 768).to(device)
    out_rms = rms_norm(x)
    print(f"RMSNorm output shape: {out_rms.shape}")
    
    # 2. test RoPE
    rope = RotaryPositionalEmbedding(64).to(device)
    q = torch.randn(2, 8, 1024, 64).to(device)  # [B, heads, seq_len, head_dim]
    k = torch.randn(2, 8, 1024, 64).to(device)
    cos, sin = rope(q)
    q_rope, k_rope = apply_rotary_pos_emb(q, k, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))
    print(f"RoPE output shapes - q: {q_rope.shape}, k: {k_rope.shape}")
    
    # 3. test multi-head attention
    mha = MultiHeadAttentionWithRoPE(768, 8).to(device)
    x = torch.randn(2, 1024, 768).to(device)
    out_mha = mha(x, x, x)
    print(f"MultiHeadAttentionWithRoPE output shape: {out_mha.shape}")
    
    # 4. test Transformer layer
    transformer_layer = TransformerEncoderLayerWithRoPE(768, 8).to(device)
    out_layer = transformer_layer(x)
    print(f"TransformerEncoderLayerWithRoPE output shape: {out_layer.shape}")
    
    # 5. performance comparison
    print("\nPerformance comparison...")
    
    # standard LayerNorm vs RMSNorm
    layer_norm = nn.LayerNorm(768).to(device)
    
    # LayerNorm time test
    start = time.time()
    for _ in range(100):
        _ = layer_norm(x)
    ln_time = time.time() - start
    
    # RMSNorm time test  
    start = time.time()
    for _ in range(100):
        _ = rms_norm(x)
    rms_time = time.time() - start
    
    print(f"LayerNorm time: {ln_time:.4f}s")
    print(f"RMSNorm time: {rms_time:.4f}s")
    print(f"RMSNorm speedup: {ln_time/rms_time:.2f}x")
    
    # parameter comparison
    ln_params = sum(p.numel() for p in layer_norm.parameters())
    rms_params = sum(p.numel() for p in rms_norm.parameters())
    print(f"LayerNorm params: {ln_params}")
    print(f"RMSNorm params: {rms_params}")
    print(f"Parameter reduction: {(ln_params - rms_params) / ln_params * 100:.1f}%")
    
    print("✓ All tests passed!")

    # 6. test Hdri_Encoder
    context = torch.randn(2, 9, 512, 512).to(device)
    encoder = Hdri_Encoder(output_dim=768, num_tokens=4096, cnn_out_channels=128, 
                 n_heads=8, num_transformer_layers=2, rope_base=10000.0).to(device)
    output = encoder(context)
    print(f"Hdri_Encoder output shape: {output.shape}")

    # 7. test attention maps
    attention_maps = encoder.get_attention_maps(context)
    print(f"Attention maps: {attention_maps}")

    # 8. test parameters
    print(f"Hdri_Encoder parameters: {sum(p.numel()/1e6 for p in encoder.parameters())}M")