import torch
import torch.nn as nn
from .core import multiscale_fft, safe_log, scale_db


def _create_frame_validity_mask(f0_hz, loudness, pitch_threshold, loudness_threshold):
    """Boolean mask of frames above pitch and loudness thresholds."""
    loudness_scaled = scale_db(loudness.squeeze(-1))
    f0_squeezed = f0_hz.squeeze(-1)
    return torch.logical_and(
        loudness_scaled > loudness_threshold,
        f0_squeezed > pitch_threshold,
    )


class MultiScaleSTFTLoss(nn.Module):
    """Multi-scale STFT loss (linear + log magnitude) for waveform reconstruction."""

    def __init__(self, scales, overlap):
        super().__init__()
        self.scales = scales
        self.overlap = overlap

    def forward(self, predicted, target):
        target_stft_mags = multiscale_fft(target, self.scales, self.overlap)
        predicted_stft_mags = multiscale_fft(predicted, self.scales, self.overlap)

        total_loss = 0.0
        for target_mag, predicted_mag in zip(target_stft_mags, predicted_stft_mags):
            linear_loss = (target_mag - predicted_mag).abs().mean()
            log_loss = (safe_log(target_mag) - safe_log(predicted_mag)).abs().mean()
            total_loss = total_loss + linear_loss + log_loss

        return total_loss


class HarmonicResidualLoss(nn.Module):
    """Harmonic Residual Loss (HRL): penalises deviation of residual corrections from 1.

    Optional temporal-smoothness penalties on (β, γ, α, B) are supported but
    disabled by default (paper uses only the residual-magnitude term).
    """

    def __init__(self, weight=0.1, weight_β=0.01, weight_α=0.01, weight_γ=0.01,
                 weight_B=0.01, weight_residuals=0.01, residual_loss_type="l1",
                 use_activation_filter: bool = True, loudness_threshold: float = 0.2,
                 pitch_threshold: float = 20.0, use_bow_mask: bool = True,
                 use_brightness_mask: bool = True, use_residuals_mask: bool = True):
        super().__init__()
        self.weight = weight
        self.weight_β = weight_β
        self.weight_α = weight_α
        self.weight_γ = weight_γ
        self.weight_B = weight_B
        self.weight_residuals = weight_residuals
        self.residual_loss_type = residual_loss_type

        self.use_activation_filter = use_activation_filter
        self.loudness_threshold = loudness_threshold
        self.pitch_threshold = pitch_threshold

        self.use_bow_mask = use_bow_mask
        self.use_brightness_mask = use_brightness_mask
        self.use_residuals_mask = use_residuals_mask

        if residual_loss_type not in ["l1", "l2"]:
            raise ValueError(f"residual_loss_type must be 'l1' or 'l2', got {residual_loss_type}")

    def forward(self, diagnostics, f0_hz=None, loudness=None):
        if diagnostics is None or not diagnostics:
            return torch.tensor(0.0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        if self.use_activation_filter:
            if f0_hz is None or loudness is None:
                raise ValueError("f0_hz and loudness must be provided for activation filtering.")
            valid_mask = _create_frame_validity_mask(f0_hz, loudness, self.pitch_threshold, self.loudness_threshold)

            if not torch.any(valid_mask):
                ref = diagnostics.get('β') or diagnostics.get('α') or diagnostics.get('residuals')
                return torch.tensor(0.0, device=ref.device)
        else:
            valid_mask = None

        total_loss = 0.0

        β = diagnostics.get('β')
        γ = diagnostics.get('γ')
        α = diagnostics.get('α')
        B = diagnostics.get('B')
        residuals = diagnostics.get('residuals')

        if self.use_bow_mask and β is not None and γ is not None:
            if β.shape[1] > 1:
                Δβ = β[:, 1:] - β[:, :-1]
                Δγ = γ[:, 1:] - γ[:, :-1]
                if valid_mask is not None:
                    valid_mask_diff = valid_mask[:, 1:] & valid_mask[:, :-1]
                    if torch.any(valid_mask_diff):
                        total_loss += self.weight_β * (Δβ[valid_mask_diff] ** 2).mean()
                        total_loss += self.weight_γ * (Δγ[valid_mask_diff] ** 2).mean()
                else:
                    total_loss += self.weight_β * (Δβ ** 2).mean()
                    total_loss += self.weight_γ * (Δγ ** 2).mean()

            if B is not None and B.shape[1] > 1:
                ΔB = B[:, 1:] - B[:, :-1]
                if valid_mask is not None:
                    valid_mask_diff = valid_mask[:, 1:] & valid_mask[:, :-1]
                    if torch.any(valid_mask_diff):
                        total_loss += self.weight_B * (ΔB[valid_mask_diff] ** 2).mean()
                else:
                    total_loss += self.weight_B * (ΔB ** 2).mean()

        if self.use_brightness_mask and α is not None:
            if α.shape[1] > 1:
                Δα = α[:, 1:] - α[:, :-1]
                if valid_mask is not None:
                    valid_mask_diff = valid_mask[:, 1:] & valid_mask[:, :-1]
                    if torch.any(valid_mask_diff):
                        total_loss += self.weight_α * (Δα[valid_mask_diff] ** 2).mean()
                else:
                    total_loss += self.weight_α * (Δα ** 2).mean()

        if self.use_residuals_mask and residuals is not None:
            deviation_from_neutral = torch.abs(residuals - 1.0)
            if self.residual_loss_type == "l2":
                deviation_from_neutral = deviation_from_neutral ** 2

            if valid_mask is not None:
                residuals_expanded = residuals.expand(-1, -1, residuals.shape[-1])
                valid_mask_expanded = valid_mask.unsqueeze(-1).expand_as(residuals_expanded)
                if torch.any(valid_mask_expanded):
                    total_loss += self.weight_residuals * deviation_from_neutral[valid_mask_expanded].mean()
            else:
                total_loss += self.weight_residuals * deviation_from_neutral.mean()

        return self.weight * total_loss
