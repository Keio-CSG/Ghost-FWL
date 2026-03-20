import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

from .FWLMAE_pretrain import Block, VoxelPatchEmbed, get_3d_sinusoid_encoding_table


class FWLMAE(nn.Module):
    """FWLMAE
    - Input: [B, 1, 100, 512, 400]
    - Output: [B, num_classes, 100, 512, 400]
    """

    def __init__(
        self,
        voxel_size: tuple = (100, 512, 400),
        patch_size: tuple = (4, 16, 16),
        in_chans: int = 1,
        num_classes: int = 4,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        fc_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        init_values: float = 0.0,
        use_learnable_pos_emb: bool = False,
        use_checkpoint: bool = False,
        use_mean_pooling: bool = False,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        # Patch embedding
        self.patch_embed = VoxelPatchEmbed(
            voxel_size=voxel_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Positional embedding
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = get_3d_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer encoder blocks (following VideoMAE)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
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

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()

        # Calculate output size per patch (patch_size * num_classes)
        patch_d, patch_h, patch_w = patch_size
        patch_output_size = patch_d * patch_h * patch_w * num_classes

        # Two-layer head to predict full patch FWL content
        self.head = (
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity(),
                nn.Linear(embed_dim // 2, patch_output_size),
            )
            if num_classes > 0
            else nn.Identity()
        )

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)

        # Initialize head weights
        if isinstance(self.head, nn.Sequential):
            for layer in self.head:
                if isinstance(layer, nn.Linear):
                    trunc_normal_(layer.weight, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self) -> int:
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {"pos_embed"}

    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: str = "") -> None:
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patch features following VideoMAE style

        Args:
            x: [B, C, D, H, W] voxel input
        Returns:
            [B, N, embed_dim] patch features
        """
        x = self.patch_embed(x)  # [B, N, embed_dim]
        B, _, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        if self.use_checkpoint:
            for blk in self.blocks:
                x = torch.utils.checkpoint.checkpoint(blk, x)  # type: ignore[attr-defined]
        else:
            for blk in self.blocks:
                x = blk(x)

        x = self.norm(x)

        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        else:
            return x  # Return all patch features [B, N, embed_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with direct patch-to-voxel prediction

        Args:
            x: [B, C, D, H, W] = [B, 1, 100, 512, 400] (same as UNet3D input)

        Returns:
            [B, num_classes, D, H, W] = [B, 4, 100, 512, 400] (same as UNet3D output)
        """
        B, C, D, H, W = x.shape

        # Extract patch features (following VideoMAE)
        patch_features = self.forward_features(x)  # [B, N, embed_dim]

        # Get per-patch predictions for full patch content (2-layer head)
        patch_predictions = self.head(
            self.fc_dropout(patch_features)
        )  # [B, N, patch_d*patch_h*patch_w*num_classes]

        # Reshape patch predictions directly to original voxel resolution
        output = self.reshape_patches_to_voxels(
            patch_predictions, (D, H, W)
        )  # [B, num_classes, D, H, W]

        return output

    def reshape_patches_to_voxels(
        self, patch_predictions: torch.Tensor, original_size: tuple
    ) -> torch.Tensor:
        """
        Reshape patch predictions directly to original voxel resolution

        Args:
            patch_predictions: [B, N, patch_d*patch_h*patch_w*num_classes]
            original_size: (D, H, W)

        Returns:
            [B, num_classes, D, H, W] - full resolution output
        """
        B, N, patch_output_size = patch_predictions.shape
        D, H, W = original_size
        patch_d, patch_h, patch_w = self.patch_size

        # Calculate patch grid dimensions
        grid_d = D // patch_d
        grid_h = H // patch_h
        grid_w = W // patch_w

        # Reshape patch predictions to [B, N, num_classes, patch_d, patch_h, patch_w]
        patch_predictions = patch_predictions.view(
            B, N, self.num_classes, patch_d, patch_h, patch_w
        )

        # Reshape to [B, grid_d, grid_h, grid_w, num_classes, patch_d, patch_h, patch_w]
        patch_predictions = patch_predictions.view(
            B, grid_d, grid_h, grid_w, self.num_classes, patch_d, patch_h, patch_w
        )

        # Rearrange to [B, num_classes, grid_d, patch_d, grid_h, patch_h, grid_w, patch_w]
        patch_predictions = patch_predictions.permute(0, 4, 1, 5, 2, 6, 3, 7)

        # Reshape to final output [B, num_classes, D, H, W]
        output = patch_predictions.contiguous().view(B, self.num_classes, D, H, W)

        return output


if __name__ == "__main__":
    model = FWLMAE(
        voxel_size=(100, 512, 400),
        patch_size=(4, 16, 16),
        in_chans=1,
        num_classes=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
    )

    # Create dummy input (same as UNet3D)
    x = torch.randn(2, 1, 100, 512, 400)

    print(f"Input shape: {x.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Expected: [B, num_classes, D, H, W] = [2, 4, 100, 512, 400]")
    print(f"Shape matches UNet3D: {output.shape == (2, 4, 100, 512, 400)}")
