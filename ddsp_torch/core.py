import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
import numpy as np
import librosa as li
import math
import os

DB_RANGE = 80.0
F0_RANGE = 127.0

_crepe_model_info_printed = False


def overlap_and_add(signal: torch.Tensor, frame_step: int) -> torch.Tensor:
    """Reconstructs a signal from overlapping frames using overlap-add."""
    if signal.dim() < 2:
        raise ValueError(f"Signal must have at least rank 2, got shape {signal.shape}")

    outer_dimensions = signal.shape[:-2]
    n_frames = signal.shape[-2]
    frame_length = signal.shape[-1]
    output_length = frame_length + frame_step * (n_frames - 1)

    if frame_length == frame_step:
        return signal.reshape(*outer_dimensions, -1)

    # Calculate padding for overlap-add processing
    segments = math.ceil(frame_length / frame_step)
    pad_length = (segments * frame_step) - frame_length
    padding = (0, pad_length, 0, segments)
    signal = F.pad(signal, padding)

    shape = outer_dimensions + (n_frames + segments, segments, frame_step)
    signal = signal.reshape(shape)

    # Permute to bring segments dimension forward for summation
    dims = list(range(signal.dim()))
    dims = dims[:-3] + [dims[-2], dims[-3], dims[-1]]
    signal = signal.permute(*dims)

    shape = outer_dimensions + ((n_frames + segments) * segments, frame_step)
    signal = signal.reshape(shape)

    # Truncate unnecessary padded frames
    signal = signal[..., :(n_frames + segments - 1) * segments, :]

    shape = outer_dimensions + (segments, n_frames + segments - 1, frame_step)
    signal = signal.reshape(shape)
    signal = torch.sum(signal, dim=-3)

    # Reshape and truncate to correct output length
    signal = signal.reshape(*outer_dimensions, -1)[..., :output_length]
    return signal


def scale_db(db: torch.Tensor) -> torch.Tensor:
    """Scales loudness in dB to range [0, 1]."""
    result = (db / DB_RANGE) + 1.0
    return torch.clamp(result, 0.0, 1.0)


def scale_f0_hz(f0_hz: torch.Tensor) -> torch.Tensor:
    """Scales frequency in Hz to range [0, 1] via MIDI conversion."""
    midi_notes = hz_to_midi(f0_hz)
    scaling_factor = torch.tensor(F0_RANGE, device=f0_hz.device, dtype=f0_hz.dtype)
    midi_notes_clamped = torch.clamp(midi_notes, 0.0, F0_RANGE)
    result = torch.div(midi_notes_clamped, scaling_factor)
    return torch.clamp(result, 0.0, 1.0)


def safe_log(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Computes log(x + eps) to avoid log(0)."""
    return torch.log(x + eps)


def multiscale_fft(signal: torch.Tensor, scales: list[int], overlap: float) -> list[torch.Tensor]:
    """Computes STFT magnitudes at multiple FFT sizes."""
    stfts = []
    signal = signal.to(torch.float32)
    
    for scale in scales:
        hop_length = int(scale * (1 - overlap))
        window = torch.hann_window(scale, device=signal.device)
        
        stft_result = torch.stft(
            signal,
            n_fft=scale,
            hop_length=hop_length,
            win_length=scale,
            window=window,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        stfts.append(torch.abs(stft_result))
    
    return stfts


def upsample_with_windows_torch(inputs: torch.Tensor, n_timesteps: int, add_endpoint: bool = True) -> torch.Tensor:
    """Upsamples control signals using Hanning windowed overlap-add."""
    inputs = inputs.float()

    if inputs.dim() != 3:
        raise ValueError(f'upsample_with_windows_torch only supports 3D input [batch, time, ch], got {inputs.shape}')

    batch_size, n_frames_orig, n_channels = inputs.shape

    if add_endpoint:
        inputs = torch.cat([inputs, inputs[:, -1:, :]], dim=1)
        n_frames = n_frames_orig + 1
    else:
        n_frames = n_frames_orig

    n_intervals = n_frames - 1
    
    if n_intervals <= 0:
        if n_timesteps % n_frames == 0:
            factor = n_timesteps // n_frames
            return inputs.repeat_interleave(factor, dim=1)
        else:
            print(f"Warning: Upsampling with 1 frame and non-divisible target length ({n_timesteps}) - tiling and cropping.")
            factor = math.ceil(n_timesteps / n_frames)
            return inputs.repeat_interleave(factor, dim=1)[:, :n_timesteps, :]

    if n_frames > n_timesteps:
        raise ValueError(f'Upsampling requires n_timesteps ({n_timesteps}) >= n_frames ({n_frames}).')

    if n_timesteps % n_intervals != 0:
        raise ValueError(f'Target n_timesteps ({n_timesteps}) must be divisible by number of intervals ({n_intervals}).')

    hop_size = n_timesteps // n_intervals
    window_length = 2 * hop_size
    window = torch.hann_window(window_length, device=inputs.device)

    # Apply windowing for overlap-add
    inputs_reshaped = inputs.permute(0, 2, 1).reshape(-1, n_frames, 1)
    windowed_inputs = inputs_reshaped * window.view(1, 1, -1)

    output = overlap_and_add(windowed_inputs, hop_size)
    output = output.reshape(batch_size, n_channels, -1).permute(0, 2, 1)

    # Trim edges from overlap-add process
    start_idx = hop_size
    end_idx = output.shape[1] - hop_size
    output_trimmed = output[:, start_idx:end_idx, :]

    # Ensure exact length
    if output_trimmed.shape[1] != n_timesteps:
        if output_trimmed.shape[1] < n_timesteps:
            pad_len = n_timesteps - output_trimmed.shape[1]
            output_trimmed = F.pad(output_trimmed, (0, 0, 0, pad_len))
        elif output_trimmed.shape[1] > n_timesteps:
            output_trimmed = output_trimmed[:, :n_timesteps, :]

    return output_trimmed


def upsample(signal: torch.Tensor, factor: int, method: str = 'linear') -> torch.Tensor:
    """Upsamples the time dimension of a signal by integer factor."""
    if factor == 1:
        return signal

    if method == 'linear':
        signal_permuted = signal.permute(0, 2, 1)
        output = nn.functional.interpolate(
            signal_permuted,
            scale_factor=float(factor),
            mode='linear',
            align_corners=False
        )
        return output.permute(0, 2, 1)
    elif method == 'window':
        target_length = signal.shape[1] * factor
        return upsample_with_windows_torch(signal, target_length)
    else:
        raise ValueError(f"Unknown upsampling method: {method}")


def remove_above_nyquist(amplitudes: torch.Tensor, pitch: torch.Tensor, sampling_rate: float | int,
                        inharmonicity_coeff: torch.Tensor | None = None) -> torch.Tensor:
    """Zeros amplitudes of harmonics above the Nyquist frequency.
    
    Args:
        amplitudes: Harmonic amplitudes [B, T, n_harmonics]
        pitch: Fundamental frequency [B, T, 1]
        sampling_rate: Audio sampling rate in Hz
        inharmonicity_coeff: Inharmonicity coefficient B [B, T, 1]. If provided,
                           applies inharmonicity before Nyquist check to avoid aliasing.
    
    Returns:
        Masked amplitudes with harmonics above Nyquist set to zero
    """
    n_harmonics = amplitudes.shape[-1]
    harmonic_numbers = torch.arange(1, n_harmonics + 1, device=amplitudes.device, dtype=amplitudes.dtype)
    
    # Calculate base harmonic frequencies
    base_harmonic_freqs = pitch * harmonic_numbers
    
    # Apply inharmonicity if provided
    if inharmonicity_coeff is not None and torch.any(inharmonicity_coeff.abs() > 1e-7):
        inharmonicity_factor = torch.sqrt(1.0 + inharmonicity_coeff * (harmonic_numbers ** 2))
        harmonic_freqs = base_harmonic_freqs * inharmonicity_factor
    else:
        harmonic_freqs = base_harmonic_freqs
    
    nyquist = sampling_rate / 2.0
    mask = (harmonic_freqs < nyquist).to(amplitudes.dtype)
    return amplitudes * mask


def exp_sigmoid(x: torch.Tensor, exponent: float = 10.0, max_value: float = 2.0, threshold: float = 1e-7) -> torch.Tensor:
    """Exponentiated sigmoid function: max_value * sigmoid(x)**log(exponent) + threshold."""
    x = x.float()
    exponent = torch.tensor(exponent, device=x.device, dtype=torch.float32)
    max_value = torch.tensor(max_value, device=x.device, dtype=torch.float32)
    threshold = torch.tensor(threshold, device=x.device, dtype=torch.float32)
    return max_value * torch.sigmoid(x)**torch.log(exponent) + threshold


def extract_loudness(signal: np.ndarray | torch.Tensor, sampling_rate: int, block_size: int,
                     n_fft: int = 2048, range_db: float = DB_RANGE, ref_db: float = 20.0) -> np.ndarray:
    """Extracts A-weighted loudness in dB using librosa."""
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()

    if signal.ndim > 1:
        print("Warning: extract_loudness received multi-channel signal, using first channel.")
        signal = signal[0] if signal.shape[0] < signal.shape[1] else signal[:, 0]

    stft_result = li.stft(signal, n_fft=n_fft, hop_length=block_size, win_length=n_fft, center=True, pad_mode='reflect')
    power_spectrum = np.abs(stft_result)**2
    frequencies = li.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    a_weighting = li.A_weighting(frequencies)

    # Apply A-weighting in dB domain
    weighted_power_db = 10.0 * np.log10(power_spectrum + 1e-10) + a_weighting[:, np.newaxis]
    loudness_db = np.mean(weighted_power_db, axis=0)

    # Apply reference and range clipping
    loudness_db -= ref_db
    loudness_db = np.maximum(loudness_db, -range_db)

    return loudness_db.astype(np.float32)


def extract_pitch(signal: np.ndarray | torch.Tensor, sampling_rate: int, block_size: int,
                  crepe_model_path: str | None = None, crepe_model_capacity: str = 'full',
                  viterbi: bool = True) -> np.ndarray:
    """Extracts fundamental frequency using CREPE."""
    global _crepe_model_info_printed
    
    try:
        import crepe
    except ImportError:
        raise ImportError("CREPE pitch extractor requires the 'crepe' library. Please install it (pip install crepe).")

    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()

    if signal.ndim > 1:
        print("Warning: extract_pitch received multi-channel signal, using first channel.")
        signal = signal[0] if signal.shape[0] < signal.shape[1] else signal[:, 0]
    
    signal = signal.astype(np.float32)
    step_size_ms = int(1000 * block_size / sampling_rate)

    # Handle custom CREPE model loading
    model_capacity_to_use = crepe_model_capacity
    if crepe_model_path and os.path.exists(crepe_model_path):
        try:
            from tensorflow.keras.models import load_model
            custom_model = load_model(crepe_model_path)
            crepe.core.models['custom'] = custom_model
            model_capacity_to_use = 'custom'
            model_info = f"custom model from {crepe_model_path}"
        except ImportError:
            print("Warning: TensorFlow/Keras not found. Cannot load custom CREPE model.")
            model_info = f"default '{model_capacity_to_use}' model (custom path ignored)"
        except Exception as e:
            print(f"Warning: Failed to load custom CREPE model from {crepe_model_path}: {e}")
            model_info = f"default '{model_capacity_to_use}' model (custom path load failed)"
    else:
        model_info = f"default '{model_capacity_to_use}' model"

    if not _crepe_model_info_printed:
        print(f"\nUsing CREPE {model_info} for pitch extraction.")
        _crepe_model_info_printed = True

    target_length = math.ceil(signal.shape[-1] / block_size)

    try:
        times, frequency, confidence, activation = crepe.predict(
            signal, sampling_rate, model_capacity=model_capacity_to_use,
            step_size=step_size_ms, viterbi=viterbi, verbose=0, center=True,
        )
    except Exception as e:
        print(f"Error during CREPE prediction: {e}")
        return np.zeros(target_length, dtype=np.float32)

    # Interpolate to match exact target length
    if len(frequency) != target_length:
        crepe_time_axis = times
        target_time_axis = np.arange(target_length) * block_size / sampling_rate
        
        if len(crepe_time_axis) < 2:
            interpolated_freq = np.full(target_length, frequency[0], dtype=np.float32) if len(frequency) == 1 else np.zeros(target_length, dtype=np.float32)
        else:
            interpolated_freq = np.interp(
                target_time_axis, crepe_time_axis, frequency,
                left=frequency[0], right=frequency[-1]
            ).astype(np.float32)
    else:
        interpolated_freq = frequency.astype(np.float32)

    # Final length adjustment
    if len(interpolated_freq) > target_length:
        interpolated_freq = interpolated_freq[:target_length]
    elif len(interpolated_freq) < target_length:
        pad_len = target_length - len(interpolated_freq)
        interpolated_freq = np.pad(interpolated_freq, (0, pad_len), mode='edge')

    return interpolated_freq


def hz_to_midi(frequencies: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Converts frequencies in Hz to MIDI notes (A4=440Hz=69)."""
    is_torch = isinstance(frequencies, torch.Tensor)
    lib = torch if is_torch else np

    min_freq = 1e-7
    valid_mask = frequencies > 0
    frequencies_safe = lib.where(valid_mask, frequencies, min_freq)

    log_term = lib.log(frequencies_safe / 440.0) / lib.log(lib.tensor(2.0) if is_torch else np.log(2.0))
    notes = 12.0 * log_term + 69.0
    output_notes = lib.where(valid_mask, notes, 0.0)

    return output_notes.float() if is_torch else output_notes.astype(np.float32)