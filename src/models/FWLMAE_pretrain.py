from functools import partial
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import trunc_normal_ as __call_trunc_normal_


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> None:
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float | None = None) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias)
            )

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float | None = None,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class VoxelPatchEmbed(nn.Module):
    """3D Voxel to Patch Embedding for LiDAR histogram data

    Converts 3D voxel data [B, C, D, H, W] to patch tokens [B, N, embed_dim]
    where D=depth(histogram), H=height(Y), W=width(X)
    """

    def __init__(
        self,
        voxel_size: tuple = (256, 128, 128),
        patch_size: tuple = (256, 16, 16),
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.voxel_size = voxel_size  # (D, H, W)
        self.patch_size = patch_size  # (patch_d, patch_h, patch_w)
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Calculate number of patches in each dimension
        self.num_patches_d = voxel_size[0] // patch_size[0]  # 100 // 4 = 25
        self.num_patches_h = voxel_size[1] // patch_size[1]  # 512 // 16 = 32
        self.num_patches_w = voxel_size[2] // patch_size[2]  # 400 // 16 = 25
        self.num_patches = (
            self.num_patches_d * self.num_patches_h * self.num_patches_w
        )  # 25 * 32 * 25 = 20000

        # 3D convolution for patch embedding
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W] = [B, 1, 100, 512, 400]
        Returns:
            [B, N, embed_dim] where N = num_patches
        """
        B, C, D, H, W = x.shape
        assert D == self.voxel_size[0] and H == self.voxel_size[1] and W == self.voxel_size[2], (
            f"Input voxel size ({D}, {H}, {W}) doesn't match model ({self.voxel_size})"
        )

        # Apply 3D convolution: [B, C, D, H, W] -> [B, embed_dim, num_patches_d, num_patches_h, num_patches_w]
        x = self.proj(x)  # [B, embed_dim, 25, 32, 25]

        # Flatten spatial dimensions and transpose: [B, embed_dim, N] -> [B, N, embed_dim]
        x = x.flatten(2).transpose(1, 2)  # [B, 20000, embed_dim]

        return x


def get_3d_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
    """3D Sinusoidal position encoding for voxel patches"""

    def get_position_angle_vec(position: int) -> list[float]:
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)


class VoxelMAEEncoder(nn.Module):
    """Voxel Vision Transformer Encoder for 3D LiDAR histogram data"""

    def __init__(
        self,
        voxel_size: tuple = (256, 128, 128),
        patch_size: tuple = (256, 16, 16),
        in_chans: int = 1,
        num_classes: int = 0,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        init_values: float | None = None,
        use_checkpoint: bool = False,
        use_learnable_pos_emb: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = VoxelPatchEmbed(
            voxel_size=voxel_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_3d_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self) -> int:
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {"pos_embed", "cls_token"}

    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: str = "") -> None:
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible

        if self.use_checkpoint:
            for blk in self.blocks:
                x_vis = checkpoint.checkpoint(blk, x_vis)
        else:
            for blk in self.blocks:
                x_vis = blk(x_vis)
        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x


class VoxelMAEDecoder(nn.Module):
    """Voxel Vision Transformer Decoder for 3D reconstruction"""

    def __init__(
        self,
        patch_size: tuple = (256, 16, 16),
        num_classes: int | None = None,
        embed_dim: int = 768,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        init_values: float | None = None,
        num_patches: int = 20000,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        # Calculate decoder output size: patch_volume * channels
        patch_volume = patch_size[0] * patch_size[1] * patch_size[2]  # 4 * 16 * 16 = 1024
        self.num_classes = num_classes or patch_volume  # Default: reconstruct voxel intensities
        self.patch_size = patch_size
        self.patch_volume = patch_volume
        self.num_features = self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self) -> int:
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {"pos_embed", "cls_token"}

    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: str = "") -> None:
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, return_token_num: int) -> torch.Tensor:
        """
        Args:
            x: [B, N, embed_dim] full sequence (visible + mask tokens)
            return_token_num: number of mask tokens to return predictions for
        Returns:
            [B, return_token_num, num_classes] predictions for masked patches
        """
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        if return_token_num > 0:
            # Only return predictions for mask tokens (last return_token_num tokens)
            x = self.head(self.norm(x[:, -return_token_num:]))
        else:
            x = self.head(self.norm(x))

        return x


class VoxelMAE(nn.Module):
    """Voxel MAE for 3D LiDAR histogram data

    Input: [B, 1, 100, 512, 400] - 3D voxel data (same as UNet3D)
    Output: [B, N_mask, patch_volume] - reconstructed masked patches
    """

    def __init__(
        self,
        voxel_size: tuple = (256, 128, 128),
        patch_size: tuple = (256, 16, 16),
        encoder_in_chans: int = 1,
        encoder_num_classes: int = 0,
        encoder_embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        decoder_num_classes: int | None = None,  # Will be set to patch_volume
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        init_values: float = 0.0,
        use_learnable_pos_emb: bool = False,
        use_checkpoint: bool = False,
        num_classes: int = 0,  # avoid error from create_fn in timm
        in_chans: int = 0,
    ) -> None:  # avoid error from create_fn in timm
        super().__init__()

        # Calculate patch volume for decoder output
        patch_volume = patch_size[0] * patch_size[1] * patch_size[2]
        if decoder_num_classes is None:
            decoder_num_classes = patch_volume

        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.patch_volume = patch_volume

        self.encoder = VoxelMAEEncoder(
            voxel_size=voxel_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_checkpoint=use_checkpoint,
            use_learnable_pos_emb=use_learnable_pos_emb,
        )

        self.decoder = VoxelMAEDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_checkpoint=use_checkpoint,
        )

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pos_embed = get_3d_sinusoid_encoding_table(
            self.encoder.patch_embed.num_patches, decoder_embed_dim
        )

        trunc_normal_(self.mask_token, std=0.02)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self) -> int:
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {"pos_embed", "cls_token", "mask_token"}

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W] = [B, 1, 100, 512, 400] voxel input (same as UNet3D)
            mask: [B, N] boolean mask where True indicates masked patches

        Returns:
            [B, N_mask, patch_volume] reconstructed values for masked patches
        """
        # Encode visible patches
        x_vis = self.encoder(x, mask)  # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]

        B, N, C = x_vis.shape

        # Prepare positional embeddings
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        # Combine visible patches with mask tokens
        x_full = torch.cat(
            [x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1
        )  # [B, N, C_d]
        # Decode masked patches
        x = self.decoder(x_full, pos_emd_mask.shape[1])  # [B, N_mask, patch_volume]
        return x

    def reconstruct_voxel(
        self, decoder_output: torch.Tensor, mask: torch.Tensor, original_shape: tuple
    ) -> torch.Tensor:
        """
        Reconstruct full voxel from decoder output

        Args:
            decoder_output: [B, N_mask, patch_volume]
            mask: [B, N] boolean mask
            original_shape: (B, C, D, H, W)

        Returns:
            [B, C, D, H, W] reconstructed voxel (same format as UNet3D output)
        """
        B, C, D, H, W = original_shape

        # Initialize output voxel
        reconstructed = torch.zeros_like(torch.empty(B, C, D, H, W, device=decoder_output.device))

        # Get patch dimensions
        patch_d, patch_h, patch_w = self.patch_size
        num_patches_d = D // patch_d
        num_patches_h = H // patch_h
        num_patches_w = W // patch_w

        # Get masked patch indices
        mask_indices = torch.where(mask)

        for i, (batch_idx, patch_idx) in enumerate(zip(mask_indices[0], mask_indices[1])):
            if i >= decoder_output.shape[1]:
                break

            # Convert patch index to 3D coordinates
            d_idx = patch_idx // (num_patches_h * num_patches_w)
            spatial_idx = patch_idx % (num_patches_h * num_patches_w)
            h_idx = spatial_idx // num_patches_w
            w_idx = spatial_idx % num_patches_w

            # Calculate voxel coordinates
            d_start, d_end = d_idx * patch_d, (d_idx + 1) * patch_d
            h_start, h_end = h_idx * patch_h, (h_idx + 1) * patch_h
            w_start, w_end = w_idx * patch_w, (w_idx + 1) * patch_w

            # Reshape patch prediction and place in voxel
            patch_pred = decoder_output[batch_idx, i].reshape(patch_d, patch_h, patch_w)
            reconstructed[batch_idx, 0, d_start:d_end, h_start:h_end, w_start:w_end] = patch_pred

        return reconstructed


class VoxelMAEPeakPositionHead(nn.Module):
    """Head to predict K peak positions per voxel (in histogram bins) for VoxelMAE."""

    def __init__(self, n_channels: int, K: int = 4, histogram_bins: int = 100) -> None:
        super().__init__()
        self.K = K
        self.histogram_bins = histogram_bins
        self.head = nn.Linear(n_channels, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, embed_dim] -> [B, N, K]
        raw_positions = self.head(x)  # (B, N, K)
        positions = torch.sigmoid(raw_positions) * (self.histogram_bins - 1)
        return positions


class VoxelMAEPeakWidthHead(nn.Module):
    """Head to predict widths (positive) for each of K peaks per voxel for VoxelMAE."""

    def __init__(self, n_channels: int, K: int = 4) -> None:
        super().__init__()
        self.head = nn.Linear(n_channels, K)
        self.activation = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, embed_dim] -> [B, N, K]
        out = self.head(x)  # (B, N, K)
        return self.activation(out)


class VoxelMAEPeakHeightHead(nn.Module):
    """Head to predict heights/amplitudes (positive) for each of K peaks per voxel for VoxelMAE."""

    def __init__(self, n_channels: int, K: int = 4) -> None:
        super().__init__()
        self.head = nn.Linear(n_channels, K)
        self.activation = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, embed_dim] -> [B, N, K]
        out = self.head(x)  # (B, N, K)
        return self.activation(out)


class FWLMAEPretrain(nn.Module):
    """
    model for peak position prediction and volume reconstruction.

    Outputs dict with keys:
    - 'peak_positions': (B, N, K) - peak positions for each patch
    - 'peak_widths': (B, N, K) - peak widths for each patch
    - 'peak_heights': (B, N, K) - peak heights for each patch
    - 'reconstruction': (B, N_mask, patch_volume) - reconstructed masked patches
    """

    def __init__(
        self,
        voxel_size: tuple = (256, 128, 128),
        patch_size: tuple = (256, 16, 16),
        encoder_in_chans: int = 1,
        encoder_num_classes: int = 0,
        encoder_embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        decoder_num_classes: int | None = None,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 8,
        K: int = 4,
        histogram_bins: int = 100,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        init_values: float = 0.0,
        use_learnable_pos_emb: bool = False,
        use_checkpoint: bool = False,
        num_classes: int = 0,
        in_chans: int = 0,
    ) -> None:
        super().__init__()

        # Calculate patch volume for decoder output
        patch_volume = patch_size[0] * patch_size[1] * patch_size[2]
        if decoder_num_classes is None:
            decoder_num_classes = patch_volume

        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.patch_volume = patch_volume
        self.K = K
        self.histogram_bins = histogram_bins

        # Base VoxelMAE model
        self.mae = VoxelMAE(
            voxel_size=voxel_size,
            patch_size=patch_size,
            encoder_in_chans=encoder_in_chans,
            encoder_num_classes=encoder_num_classes,
            encoder_embed_dim=encoder_embed_dim,
            encoder_depth=encoder_depth,
            encoder_num_heads=encoder_num_heads,
            decoder_num_classes=decoder_num_classes,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb,
            use_checkpoint=use_checkpoint,
            num_classes=num_classes,
            in_chans=in_chans,
        )

        # Peak prediction heads on encoder features
        self.peak_position_head = VoxelMAEPeakPositionHead(
            decoder_embed_dim, K=K, histogram_bins=histogram_bins
        )
        self.peak_width_head = VoxelMAEPeakWidthHead(decoder_embed_dim, K=K)
        self.peak_height_head = VoxelMAEPeakHeightHead(decoder_embed_dim, K=K)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, D, H, W] voxel input
            mask: [B, N] boolean mask where True indicates masked patches

        Returns:
            Dictionary containing:
            - 'peak_positions': (B, N, K) - peak positions for all patches
            - 'peak_widths': (B, N, K) - peak widths for all patches
            - 'peak_heights': (B, N, K) - peak heights for all patches
            - 'reconstruction': (B, N_mask, patch_volume) - reconstructed masked patches
        """
        # Get encoder features for peak prediction
        x_vis = self.mae.encoder(x, mask)  # [B, N_vis, embed_dim]
        x_vis = self.mae.encoder_to_decoder(x_vis)  # [B, N_vis, embed_dim]

        # Get full sequence for peak prediction (visible + masked patches)
        B, N, C = x_vis.shape
        expand_pos_embed = (
            self.mae.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        )
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)

        # Combine visible patches with mask tokens for full sequence
        x_full = torch.cat(
            [x_vis + pos_emd_vis, self.mae.mask_token + pos_emd_mask], dim=1
        )  # [B, N, embed_dim]

        # Peak predictions from full sequence
        result: Dict[str, torch.Tensor] = {}
        result["peak_positions"] = self.peak_position_head(x_full)  # (B, N, K)
        result["peak_widths"] = self.peak_width_head(x_full)  # (B, N, K)
        result["peak_heights"] = self.peak_height_head(x_full)  # (B, N, K)

        # MAE reconstruction
        result["reconstruction"] = self.mae.decoder(
            x_full, pos_emd_mask.shape[1]
        )  # (B, N_mask, patch_volume)

        return result

    def reconstruction_only_forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that only computes reconstruction, skipping peak prediction heads.
        This is more efficient when only reconstruction evaluation is needed.

        Args:
            x: Input tensor (B, C, D, H, W)
            mask: Boolean mask (B, N)

        Returns:
            Reconstructed tensor (B, N_mask, patch_volume)
        """
        return self.mae(x, mask)

    def reconstruct_voxel(
        self, decoder_output: torch.Tensor, mask: torch.Tensor, original_shape: tuple
    ) -> torch.Tensor:
        """
        Reconstruct full voxel from decoder output using the base MAE method.

        Args:
            decoder_output: [B, N_mask, patch_volume]
            mask: [B, N] boolean mask
            original_shape: (B, C, D, H, W)

        Returns:
            [B, C, D, H, W] reconstructed voxel
        """
        return self.mae.reconstruct_voxel(decoder_output, mask, original_shape)
