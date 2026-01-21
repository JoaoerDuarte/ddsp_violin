import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional

from .dsp import fft_convolve, frequency_to_impulse_response
from .core import upsample, exp_sigmoid


def calculate_inharmonic_frequencies(pitch_frames: torch.Tensor, inharmonicity_coeff_frames: torch.Tensor,
                                   n_harmonic: int, sampling_rate: float) -> torch.Tensor:
    """Calculates inharmonic frequencies using Railsback's formula."""
    device, dtype = pitch_frames.device, pitch_frames.dtype
    harmonic_numbers = torch.arange(1, n_harmonic + 1, device=device, dtype=dtype).view(1, 1, -1)
    
    pitch_expanded = pitch_frames.unsqueeze(-1) if pitch_frames.dim() == 2 else pitch_frames
    inharmonicity_expanded = inharmonicity_coeff_frames.unsqueeze(-1) if inharmonicity_coeff_frames.dim() == 2 else inharmonicity_coeff_frames
    
    inharmonicity_factor = torch.sqrt(1.0 + inharmonicity_expanded * (harmonic_numbers ** 2))
    base_harmonic_freqs = pitch_expanded * harmonic_numbers
    inharmonic_freqs = base_harmonic_freqs * inharmonicity_factor
    
    return inharmonic_freqs


def harmonic_synth(pitch_frames: torch.Tensor, amplitudes_dist_normalized_frames: torch.Tensor, 
                   total_amp_frames: torch.Tensor, sampling_rate: int, block_size: int,
                   inharmonicity_coeff_frames: torch.Tensor | None = None):
    """Generates harmonic signal using additive synthesis (standard DDSP mode)."""
    n_harmonic = amplitudes_dist_normalized_frames.shape[-1]
    
    total_amplitude_expanded = total_amp_frames.unsqueeze(-1) if total_amp_frames.dim() == 2 else total_amp_frames
    final_amplitudes = amplitudes_dist_normalized_frames * total_amplitude_expanded
    amplitudes_upsampled = upsample(final_amplitudes, block_size, method='window')

    if inharmonicity_coeff_frames is not None and torch.any(inharmonicity_coeff_frames.abs() > 1e-7):
        frequencies_frames = calculate_inharmonic_frequencies(
            pitch_frames, inharmonicity_coeff_frames, n_harmonic, float(sampling_rate)
        )
    else:
        device, dtype = pitch_frames.device, pitch_frames.dtype
        harmonic_numbers = torch.arange(1, n_harmonic + 1, device=device, dtype=dtype).view(1, 1, -1)
        pitch_expanded = pitch_frames.unsqueeze(-1) if pitch_frames.dim() == 2 else pitch_frames
        frequencies_frames = pitch_expanded * harmonic_numbers
    
    frequencies_upsampled = upsample(frequencies_frames, block_size, method='window')
    
    phase = torch.cumsum(2 * math.pi * frequencies_upsampled / float(sampling_rate), dim=1)
    harmonic_signals = torch.sin(phase) * amplitudes_upsampled
    signal = harmonic_signals.sum(dim=-1, keepdim=True)
    
    return signal


def filtered_noise_synth(noise_magnitudes_frames: torch.Tensor, block_size: int):
    """Generates filtered noise using frequency-domain filtering."""
    batch_size, n_time_frames, n_bands = noise_magnitudes_frames.shape
    
    ir_size = n_bands * 2 - 1
    impulse_responses = frequency_to_impulse_response(noise_magnitudes_frames, window_size=ir_size)
    
    total_samples = n_time_frames * block_size
    noise_signal = torch.rand(batch_size, total_samples, device=impulse_responses.device) * 2 - 1
    
    filtered_noise = fft_convolve(noise_signal, impulse_responses, padding='same')
    
    if filtered_noise.dim() == 2:
        filtered_noise = filtered_noise.unsqueeze(-1)
    
    if filtered_noise.shape[1] > total_samples:
        filtered_noise = filtered_noise[:, :total_samples, :]
    elif filtered_noise.shape[1] < total_samples:
        pad_amount = total_samples - filtered_noise.shape[1]
        filtered_noise = F.pad(filtered_noise, (0, 0, 0, pad_amount))
    
    return filtered_noise