#!/usr/bin/env python3
"""
Multi-task loss function for peak prediction model.

This loss function handles the combined training of:
- Peak position prediction
- Peak height prediction
- Peak width prediction
- Volume reconstruction

Each component can be weighted differently to balance the multi-task learning.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FWLMAELoss(nn.Module):
    """
    Patch-space (B, N, K) based loss for joint FWL reconstruction and peak regression.

    Args:
        - reconstruction: (B, N_mask, patch_volume) - FWL reconstruction
        - peak_*:         (B, N, K)  # Full sequence (visible → masked order, mask last) - Peak positions, heights, widths
        - masks:          (B, N), bool, True = masked - Masked patches
    """

    def __init__(
        self,
        patch_size: tuple = (256, 16, 16),
        position_weight: float = 1.0,
        height_weight: float = 1.0,
        width_weight: float = 0.5,
        mae_reconstruction_weight: float = 1.0,
        position_loss: str = "l1",
        height_loss: str = "l1",
        width_loss: str = "l1",
        mae_reconstruction_loss: str = "mse",
        use_valid_mask: bool = True,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.position_weight = position_weight
        self.height_weight = height_weight
        self.width_weight = width_weight
        self.mae_reconstruction_weight = mae_reconstruction_weight
        self.use_valid_mask = use_valid_mask
        self.epsilon = epsilon

        self.position_loss_fn = self._get_loss_fn(position_loss)
        self.height_loss_fn = self._get_loss_fn(height_loss)
        self.width_loss_fn = self._get_loss_fn(width_loss)
        self.mae_reconstruction_loss_fn = self._get_loss_fn(mae_reconstruction_loss)

    def _get_loss_fn(self, loss_type: str) -> nn.Module:
        loss_type = loss_type.lower()
        if loss_type == "mse":
            return nn.MSELoss(reduction="none")
        elif loss_type == "l1":
            return nn.L1Loss(reduction="none")
        elif loss_type == "smooth_l1":
            return nn.SmoothL1Loss(reduction="none")
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    @torch.no_grad()
    def _patchify_targets_bkHW_to_bnK(self, peak_x: torch.Tensor) -> torch.Tensor:
        """
        Patchify targets from (B,K,H,W) to (B,N,K)
        For peak data, we take the maximum value in each patch to preserve peak information.
        """
        B, K, H, W = peak_x.shape
        ph, pw = self.patch_size[1], self.patch_size[2]  # patch height and width

        # Calculate number of patches in H and W dimensions
        num_patches_h = H // ph
        num_patches_w = W // pw
        num_patches = num_patches_h * num_patches_w

        # Reshape to patches: (B, K, H, W) -> (B, K, num_patches_h, ph, num_patches_w, pw)
        peak_x = peak_x.view(B, K, num_patches_h, ph, num_patches_w, pw)

        # Transpose and reshape to (B, K, num_patches, ph*pw)
        peak_x = peak_x.permute(
            0, 1, 2, 4, 3, 5
        ).contiguous()  # (B, K, num_patches_h, num_patches_w, ph, pw)
        peak_x = peak_x.view(B, K, num_patches, ph * pw)
        # Alternatively, take the mean over patch spatial dimensions within each patch (as allowed)
        peak_x = peak_x.mean(dim=-1)  # (B, K, num_patches)

        # Transpose to (B, num_patches, K)
        peak_x = peak_x.transpose(1, 2)  # (B, num_patches, K)

        return peak_x

    def _compute_mae_recon_loss(
        self,
        reconstruction: torch.Tensor,  # (B, N_mask, patch_volume)
        target_patches: torch.Tensor,  # (B, N, patch_volume)
        masks: torch.Tensor,  # (B, N), bool
    ) -> torch.Tensor:
        """
        Computes patch MAE loss for masked patches only.
        """
        # Gather target patches for masked locations
        target_masked = target_patches[masks]  # (sum_b N_mask_b, patch_volume)
        recon_flat = reconstruction.reshape(-1, reconstruction.size(-1))
        target_flat = target_masked.reshape(-1, target_patches.size(-1))
        loss = self.mae_reconstruction_loss_fn(recon_flat, target_flat)  # (numel,)
        return loss.mean()

    def _compute_peak_losses(
        self,
        predictions: Dict[
            str, torch.Tensor
        ],  # 'peak_positions','peak_heights','peak_widths' : (B,N,K)
        targets: Dict[str, torch.Tensor],  # (B,K,H,W) - need to be patchified
        masks: torch.Tensor,  # (B,N), bool (True=masked)
    ) -> Dict[str, torch.Tensor]:
        """
        Computes regression losses for peaks on masked patches.
        """

        # Patchify targets from (B,K,H,W) to (B,N,K)
        tgt_pos_patched = self._patchify_targets_bkHW_to_bnK(targets["peak_positions"])
        tgt_hgt_patched = self._patchify_targets_bkHW_to_bnK(targets["peak_heights"])
        tgt_wid_patched = self._patchify_targets_bkHW_to_bnK(targets["peak_widths"])

        # Extract predictions and targets for masked patches
        pred_pos_m = predictions["peak_positions"][masks]
        pred_hgt_m = predictions["peak_heights"][masks]
        pred_wid_m = predictions["peak_widths"][masks]

        tgt_pos_m = tgt_pos_patched[masks]
        tgt_hgt_m = tgt_hgt_patched[masks]
        tgt_wid_m = tgt_wid_patched[masks]

        # Handle empty case (no masked patches)
        if pred_pos_m.numel() == 0:
            zero = next(self.parameters()).new_zeros(())
            return {
                "position_loss": zero,
                "height_loss": zero,
                "width_loss": zero,
            }

        # Individual losses (mean over all masked elements)
        pos_loss = self.position_loss_fn(pred_pos_m, tgt_pos_m).mean()
        hgt_loss = self.height_loss_fn(pred_hgt_m, tgt_hgt_m).mean()
        wid_loss = self.width_loss_fn(pred_wid_m, tgt_wid_m).mean()

        return {
            "position_loss": pos_loss,
            "height_loss": hgt_loss,
            "width_loss": wid_loss,
        }

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],  # 'reconstruction':(B,N_mask,P), 'peak_*':(B,N,K)
        targets: Dict[str, torch.Tensor],  # 'peak_*':(B,K,H,W)
        masks: torch.Tensor,  # (B,N), bool
        target_patches: torch.Tensor,  # (B,N,P)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the total loss and each loss component for MAE+peak prediction in patch space.

        Args:
            predictions: Model output dictionary containing 'reconstruction' and peak_* keys.
            targets:     Dict with ground truth for peak_* keys.
            masks:       Patch mask (True = masked).
            target_patches: The target voxel patches (B,N,P) for MAE.
        Returns:
            total_loss:      Weighted sum of all loss components.
            loss_components: Dictionary with individual component losses.
        """
        peak_losses = self._compute_peak_losses(predictions, targets, masks)
        mae_loss = self._compute_mae_recon_loss(
            reconstruction=predictions["reconstruction"],
            target_patches=target_patches,
            masks=masks,
        )

        total_loss = (
            self.position_weight * peak_losses["position_loss"]
            + self.height_weight * peak_losses["height_loss"]
            + self.width_weight * peak_losses["width_loss"]
            + self.mae_reconstruction_weight * mae_loss
        )

        loss_components = {
            "position_loss": peak_losses["position_loss"],
            "height_loss": peak_losses["height_loss"],
            "width_loss": peak_losses["width_loss"],
            "mae_reconstruction_loss": mae_loss,
            "total_loss": total_loss,
        }
        return total_loss, loss_components


if __name__ == "__main__":
    # Test the loss function
    print("=== Testing PeakPredictionLoss ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create test data
    batch_size = 2
    K = 4  # max peaks
    D, H, W = 50, 64, 64  # depth, height, width
    C = 1  # channels

    # Model predictions
    predictions = {
        "peak_positions": torch.randn(batch_size, K, D, H, W, device=device),
        "peak_heights": torch.randn(batch_size, K, D, H, W, device=device).abs(),
        "peak_widths": torch.randn(batch_size, K, D, H, W, device=device).abs(),
        "reconstruction": torch.randn(batch_size, C, D, H, W, device=device).abs(),
    }

    # Targets
    targets = {
        "peak_positions": torch.randn(batch_size, K, H, W, device=device) * 50,  # 0-49 range
        "peak_heights": torch.randn(batch_size, K, H, W, device=device).abs(),
        "peak_widths": torch.randn(batch_size, K, H, W, device=device).abs(),
        "voxels": torch.randn(batch_size, C, D, H, W, device=device).abs(),
    }

    # Valid masks (some peaks are invalid)
    valid_masks = torch.rand(batch_size, K, H, W, device=device) > 0.3

    # Test standard loss
    loss_fn = PeakPredictionLoss().to(device)
    total_loss, loss_components = loss_fn(predictions, targets, valid_masks)

    print(f"Total loss: {total_loss.item():.6f}")
    print("Loss components:")
    for name, value in loss_components.items():
        if isinstance(value, torch.Tensor):
            print(f"  {name}: {value.item():.6f}")
