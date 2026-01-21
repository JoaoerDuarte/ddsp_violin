import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import math
from .core import overlap_and_add


def apply_window_to_impulse_response(impulse_response: torch.Tensor, window_size: int = 0, causal: bool = False) -> torch.Tensor:
    """Applies a Hann window to an impulse response, handling causality and centering."""
    if causal:
        impulse_response = torch.fft.fftshift(impulse_response, dim=-1)

    ir_size = impulse_response.shape[-1]
    if window_size <= 0 or window_size > ir_size:
        window_size = ir_size
    
    window = torch.hann_window(window_size, dtype=impulse_response.dtype, device=impulse_response.device)

    padding = ir_size - window_size
    if padding > 0:
        half_idx = (window_size + 1) // 2
        # Pad window symmetrically in zero-phase form
        window = torch.cat([
            window[half_idx:],
            torch.zeros(padding, device=window.device, dtype=window.dtype),
            window[:half_idx]
        ])
    else:
        window = torch.fft.fftshift(window, dim=-1)

    windowed_ir = window * impulse_response.real

    if causal:
        if padding > 0:
            half_idx = (window_size + 1) // 2
            first_half_start = ir_size - (half_idx - 1)
            second_half_end = half_idx
            # Reconstruct causal form from zero-phase windowed signal
            windowed_ir = torch.cat([
                windowed_ir[..., first_half_start:],
                windowed_ir[..., :second_half_end]
            ], dim=-1)
        else:
            windowed_ir = torch.fft.fftshift(windowed_ir, dim=-1)

    return windowed_ir


def frequency_to_impulse_response(magnitude_response: torch.Tensor, window_size: int = 0) -> torch.Tensor:
    """Converts frequency magnitudes to a time-domain impulse response via iFFT."""
    complex_spectrum = torch.view_as_complex(
        torch.stack([magnitude_response, torch.zeros_like(magnitude_response)], -1)
    )
    impulse_response = torch.fft.irfft(complex_spectrum)

    # Window and convert to causal form
    impulse_response = apply_window_to_impulse_response(
        impulse_response, window_size=window_size, causal=True
    )
    return impulse_response


def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True) -> int:
    """Calculates required FFT size for convolution to avoid circular convolution."""
    convolved_frame_size = ir_size + frame_size - 1
    if power_of_2:
        return 2**math.ceil(math.log2(convolved_frame_size))
    return convolved_frame_size


def crop_and_compensate_delay(audio: torch.Tensor, audio_size: int, ir_size: int,
                             padding: str = 'same', delay_compensation: int = -1) -> torch.Tensor:
    """Crops convolved audio to specified size and compensates for filter delay."""
    if padding == 'valid':
        crop_size = ir_size + audio_size - 1
    elif padding == 'same':
        crop_size = audio_size
    else:
        raise ValueError(f'Padding must be valid or same, got {padding}')

    total_size = audio.shape[-1]
    crop_amount = total_size - crop_size

    if delay_compensation < 0:
        start = (ir_size - 1) // 2
    else:
        start = delay_compensation

    end = crop_amount - start

    # Ensure valid indices
    start = max(0, start)
    end = max(0, end)
    end_index = total_size - end

    if start >= end_index:
        print(f"Warning: crop_and_compensate_delay resulted in empty slice, returning original.")
        return audio

    return audio[..., start:end_index]


def fft_convolve(audio: torch.Tensor, impulse_response: torch.Tensor,
                 padding: str = 'same', delay_compensation: int = -1) -> torch.Tensor:
    """Performs FFT-based convolution using overlap-add for time-varying IRs."""
    batch_size, audio_size = audio.shape
    ir_shape = impulse_response.shape

    if len(ir_shape) == 2:
        impulse_response = impulse_response.unsqueeze(0)

    ir_batch_size = impulse_response.shape[0]
    if ir_batch_size == 1 and batch_size > 1:
        impulse_response = impulse_response.expand(batch_size, -1, -1)
    elif ir_batch_size > 1 and ir_batch_size != batch_size:
        raise ValueError(f"Batch dimension mismatch: audio ({batch_size}) vs IR ({ir_batch_size})")

    ir_size = impulse_response.shape[-1]
    n_ir_frames = impulse_response.shape[-2]

    # Frame audio to match the number of IR frames
    frame_size = math.ceil(audio_size / n_ir_frames)
    total_audio_samples_needed = n_ir_frames * frame_size
    pad_size = total_audio_samples_needed - audio_size
    audio_padded = F.pad(audio, (0, pad_size))

    audio_frames = audio_padded.unfold(-1, frame_size, frame_size)
    if audio_frames.shape[1] != n_ir_frames:
        print(f"Warning: Frame mismatch in fft_convolve audio ({audio_frames.shape[1]}) vs IR ({n_ir_frames}).")

    fft_size = get_fft_size(frame_size, ir_size, power_of_2=True)

    audio_fft = torch.fft.rfft(audio_frames, n=fft_size)
    ir_fft = torch.fft.rfft(impulse_response, n=fft_size)

    # Convolution in frequency domain
    convolved_fft = audio_fft * ir_fft
    audio_frames_out = torch.fft.irfft(convolved_fft, n=fft_size)

    # Overlap-add the convolved frames
    output = overlap_and_add(audio_frames_out, frame_size)

    return crop_and_compensate_delay(output, audio_size, ir_size, padding, delay_compensation)