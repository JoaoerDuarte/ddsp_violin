import torch
import torch.nn as nn
from .core import multiscale_fft, safe_log, hz_to_midi, scale_db

def _create_frame_validity_mask(f0_hz, loudness, pitch_threshold, loudness_threshold):
    """Creates a boolean mask to select frames that are above pitch and loudness thresholds."""
    loudness_scaled = scale_db(loudness.squeeze(-1))
    f0_squeezed = f0_hz.squeeze(-1)
    
    valid_mask = torch.logical_and(
        loudness_scaled > loudness_threshold,
        f0_squeezed > pitch_threshold
    )
    return valid_mask

class MultiScaleSTFTLoss(nn.Module):
    """Multi-scale STFT loss for perceptual audio reconstruction."""
    
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

class SourceVarianceLoss(nn.Module):
    """Variance regularization loss for harmonic amplitudes in standard DDSP mode."""
    
    def __init__(self, weight=1.0, loudness_threshold=0.2, pitch_threshold=20.0,
                 distribution_type='var', power_type='pwr', per_pitch=False,
                 normalize_metrics=True, epsilon=1e-8, n_harmonics=10):
        super().__init__()
        self.weight = weight
        self.loudness_threshold = loudness_threshold
        self.pitch_threshold = pitch_threshold
        self.distribution_type = distribution_type
        self.power_type = power_type
        self.per_pitch = per_pitch
        self.normalize_metrics = normalize_metrics
        self.epsilon = epsilon
        self.n_harmonics = n_harmonics

    def forward(self, harmonic_distribution, f0_hz, loudness):
        if harmonic_distribution is None:
            device = self._get_device(f0_hz, loudness)
            return torch.tensor(0.0, device=device)

        valid_mask = _create_frame_validity_mask(f0_hz, loudness, self.pitch_threshold, self.loudness_threshold)
        
        valid_harmonics = harmonic_distribution[valid_mask]

        if valid_harmonics.numel() == 0:
            return torch.tensor(0.0, device=harmonic_distribution.device)

        if self.n_harmonics is not None and self.n_harmonics < valid_harmonics.shape[-1]:
            valid_harmonics = valid_harmonics[:, :self.n_harmonics]

        if self.power_type == 'pwr':
            harmonics_for_metric = valid_harmonics.pow(2)
        else:
            harmonics_for_metric = valid_harmonics

        if self.distribution_type == 'std':
            metric = torch.std(harmonics_for_metric, dim=0)
        else:
            metric = torch.var(harmonics_for_metric, dim=0)

        if self.normalize_metrics:
            mean_values = torch.mean(harmonics_for_metric, dim=0)
            if self.distribution_type == 'var':
                metric = metric / ((mean_values + self.epsilon) ** 2)
            else:
                metric = metric / (mean_values + self.epsilon)

        loss = metric.sum()
        return self.weight * loss

    def _get_device(self, f0_hz, loudness):
        if f0_hz is not None: return f0_hz.device
        elif loudness is not None: return loudness.device
        else: return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ViolinPhysicsLoss(nn.Module):
    """Regularization loss for violin mode physics parameters."""
    
    def __init__(self, weight=0.1, weight_β=0.01, weight_α=0.01, weight_γ=0.01, 
                 weight_B=0.01, weight_residuals=0.01, residual_loss_type="l1",
                 use_activation_filter: bool = True, loudness_threshold: float = 0.2, 
                 pitch_threshold: float = 20.0):
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
                return torch.tensor(0.0, device=diagnostics['β'].device)
        else:
            valid_mask = None
        
        total_loss = 0.0
        
        β = diagnostics['β']
        γ = diagnostics['γ']
        α = diagnostics['α']
        B = diagnostics['B']
        residuals = diagnostics['residuals']
        
        if β.shape[1] > 1:
            Δβ = β[:, 1:] - β[:, :-1]
            Δα = α[:, 1:] - α[:, :-1]
            Δγ = γ[:, 1:] - γ[:, :-1]
            ΔB = B[:, 1:] - B[:, :-1]
            
            if valid_mask is not None:
                valid_mask_diff = valid_mask[:, 1:] & valid_mask[:, :-1]
                
                if torch.any(valid_mask_diff):
                    total_loss += self.weight_β * (Δβ[valid_mask_diff] ** 2).mean()
                    total_loss += self.weight_α * (Δα[valid_mask_diff] ** 2).mean()
                    total_loss += self.weight_γ * (Δγ[valid_mask_diff] ** 2).mean()
                    total_loss += self.weight_B * (ΔB[valid_mask_diff] ** 2).mean()
            else:
                total_loss += self.weight_β * (Δβ ** 2).mean()
                total_loss += self.weight_α * (Δα ** 2).mean()
                total_loss += self.weight_γ * (Δγ ** 2).mean()
                total_loss += self.weight_B * (ΔB ** 2).mean()
        
        deviation_from_neutral = torch.abs(residuals - 1.0)
        
        if self.residual_loss_type == "l2":
            deviation_from_neutral = deviation_from_neutral ** 2
        
        if valid_mask is not None:
            residuals_expanded = residuals.expand(-1, -1, residuals.shape[-1])
            valid_mask_expanded = valid_mask.unsqueeze(-1).expand_as(residuals_expanded)
            
            if torch.any(valid_mask_expanded):
                residual_loss = deviation_from_neutral[valid_mask_expanded].mean()
                total_loss += self.weight_residuals * residual_loss
        else:
            total_loss += self.weight_residuals * deviation_from_neutral.mean()
        
        return self.weight * total_loss