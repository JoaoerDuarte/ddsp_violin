import torch
from typing import Dict, Any

# Dictionary keys shared between modules
F0_SCALED = 'f0_scaled'
LD_SCALED = 'ld_scaled'
Z = 'z'
AMPS = 'amps'
HARMONIC_DISTRIBUTION = 'harmonic_distribution'
NOISE_MAGNITUDES = 'noise_magnitudes'
BOW_POSITION_RAW = 'bow_position_raw'
NOTCH_DEPTH_RAW = 'notch_depth_raw'
BRIGHTNESS_RAW = 'brightness_raw'
RESIDUALS_RAW = 'residuals_raw'
INHARMONICITY_COEFF = 'inharmonicity_coeff'


def count_parameters(model):
    """Counts the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_filter(filter_name: str, filter_type: str, config: Dict[str, Any]):
    """Creates a filter of the specified type with proper error handling."""
    from .filters import ConvolutionalFilter, ARFilter, ARMAFilter
    
    is_room = filter_name == "room"
    
    try:
        if filter_type == "ar":
            return ARFilter(**extract_filter_config(config, is_room=is_room))
        elif filter_type == "arma":
            return ARMAFilter(**extract_filter_config(config, is_room=is_room))
        elif filter_type == "convolutional":
            is_resonance = not is_room
            return ConvolutionalFilter(**extract_conv_config(config, is_resonance=is_resonance))
        else:
            print(f"Warning: Unknown {filter_name} filter type '{filter_type}', skipping.")
            return None
    except Exception as e:
        print(f"Warning: Failed to initialize {filter_name} filter '{filter_type}': {e}")
        return None


def extract_conv_config(config: Dict[str, Any], is_resonance: bool = True) -> Dict[str, Any]:
    """Extracts convolutional filter configuration with role-based defaults."""
    prefix = "resonance_" if is_resonance else "room_"
    
    if is_resonance:
        default_add_dry = False
        default_mask_ir = False
        default_normalization = "psd_weighted_rms"
    else:
        default_add_dry = True
        default_mask_ir = True
        default_normalization = "psd_weighted_rms"
    
    normalization_type = config.get(f"{prefix}normalization_type", default_normalization)
    
    conv_config = {
        "length": int(config.get(f"{prefix}length", 48000)),
        "add_dry": bool(config.get(f"{prefix}add_dry", default_add_dry)),
        "mask_ir": bool(config.get(f"{prefix}mask_ir", default_mask_ir)),
        "normalization_type": normalization_type
    }
    
    if normalization_type in ["psd_weighted_rms"]:
        conv_config["n_fft"] = int(config.get(f"{prefix}n_fft", 2048))
        conv_config["psd_prior"] = str(config.get(f"{prefix}psd_prior", "pink"))
        conv_config["psd_ema_alpha"] = float(config.get(f"{prefix}psd_ema_alpha", 0.99))
    
    return conv_config


def extract_filter_config(config: Dict[str, Any], is_room: bool = False) -> Dict[str, Any]:
    """Extracts AR/ARMA filter configuration with role-based defaults."""
    prefix = "room_" if is_room else "resonance_"
    
    if is_room:
        default_add_dry = True
        default_normalization = "rms"
    else:
        default_add_dry = False
        default_normalization = "rms"
    
    filter_config = {
        "add_dry": bool(config.get(f"{prefix}add_dry", default_add_dry)),
        "max_reflection": float(config.get(f"{prefix}max_reflection", 0.999)),
        "normalization_type": config.get(f"{prefix}normalization_type", default_normalization)
    }
    
    if config.get(f"{prefix}type") in ["ar", "resonance_type"] and config.get("resonance_type") == "ar":
        filter_config["lpc_order"] = int(config.get(f"{prefix}ar_order", 16))
        filter_config["window_size"] = int(config.get(f"{prefix}window_size", 4096))
    elif config.get(f"{prefix}type") == "ar" or (not is_room and config.get("resonance_type") == "ar"):
        filter_config["lpc_order"] = int(config.get(f"{prefix}ar_order", 16))
        filter_config["window_size"] = int(config.get(f"{prefix}window_size", 4096))
    
    if config.get(f"{prefix}type") == "arma" or (is_room and config.get("room_type") == "arma") or (not is_room and config.get("resonance_type") == "arma"):
        filter_config["ar_order"] = int(config.get(f"{prefix}ar_order", 16))
        filter_config["ma_order"] = int(config.get(f"{prefix}ma_order", 16))
        filter_config["window_size"] = int(config.get(f"{prefix}window_size", 4096))
    
    return filter_config


def get_inharmonicity_coefficient(synthesis_params: Dict[str, torch.Tensor], pitch: torch.Tensor,
                                 use_inharmonicity: bool, inharmonicity_b_max: float = 0.0005) -> torch.Tensor:
    """Extracts or creates inharmonicity coefficient B."""
    if use_inharmonicity and INHARMONICITY_COEFF in synthesis_params:
        B_raw = synthesis_params[INHARMONICITY_COEFF]
        B = torch.sigmoid(B_raw) * inharmonicity_b_max
    else:
        B = torch.zeros_like(pitch)
    return B


def print_model_summary(encoder, decoder, resonance, room, use_helmholtz, violin_config, use_inharmonicity):
    """Prints a formatted summary of the model architecture and parameters."""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    
    if encoder is not None:
        encoder_params = count_parameters(encoder)
        print(f"\nEncoder: {encoder_params:,} parameters")
    else:
        print("\nEncoder: Not used")
    
    decoder_params = count_parameters(decoder)
    print(f"Decoder: {decoder_params:,} parameters")
    
    if resonance is not None:
        resonance_params = count_parameters(resonance)
        resonance_type = resonance.__class__.__name__
        print(f"Resonance Filter ({resonance_type}): {resonance_params:,} parameters")
    else:
        print("Resonance Filter: Not used")
    
    if room is not None:
        room_params = count_parameters(room)
        room_type = room.__class__.__name__
        print(f"Room Filter ({room_type}): {room_params:,} parameters")
    else:
        print("Room Filter: Not used")
    
    total_params = decoder_params
    if encoder is not None:
        total_params += encoder_params
    if resonance is not None:
        total_params += resonance_params
    if room is not None:
        total_params += room_params
    
    print(f"\nTotal Parameters: {total_params:,}")
    
    print("\n" + "-"*60)
    print("SYNTHESIS CONFIGURATION")
    print("-"*60)
    
    if use_helmholtz:
        print("\nSynthesis Mode: Violin (Physics-Guided)")
        print(f"- Bow Position Range: [{violin_config.get('β_min', 'N/A')}, {violin_config.get('β_max', 'N/A')}]")
        print(f"- Brightness Range: [{violin_config.get('α_min', 'N/A')}, {violin_config.get('α_max', 'N/A')}]")
        print(f"- Notch Width: {violin_config.get('notch_width', 'N/A')}")
        print(f"- Residuals: {violin_config.get('n_residuals', 'N/A')}")
        print(f"- Residual Scale: {violin_config.get('residual_scale', 'N/A')}")
    else:
        print("\nSynthesis Mode: Standard DDSP")
    
    if use_inharmonicity:
        print("- Inharmonicity: Enabled")
    else:
        print("- Inharmonicity: Disabled")
    
    print("="*60 + "\n")


def activate_bow_position(raw: torch.Tensor, β_min: float, β_max: float) -> torch.Tensor:
    """Maps raw decoder output to bow position β in [β_min, β_max]."""
    sigmoid_val = torch.sigmoid(raw)
    return β_min + (β_max - β_min) * sigmoid_val


def activate_notch_depth(raw: torch.Tensor) -> torch.Tensor:
    """Maps raw decoder output to notch depth γ in [0, 1]."""
    return torch.sigmoid(raw)


def activate_brightness(raw: torch.Tensor, α_min: float, α_max: float) -> torch.Tensor:
    """Maps raw decoder output to brightness α in [α_min, α_max]."""
    sigmoid_val = torch.sigmoid(raw)
    return α_min + (α_max - α_min) * sigmoid_val


def activate_residuals(raw: torch.Tensor, residual_scale: float) -> torch.Tensor:
    """Maps raw decoder output to residuals centered at 1.0."""
    tanh_val = torch.tanh(raw)
    return 1.0 + residual_scale * tanh_val


def compute_bow_notch(β: torch.Tensor, n_harmonic: int) -> torch.Tensor:
    """Computes ideal Helmholtz bow notch for given bow position."""
    device = β.device
    dtype = β.dtype
    batch_size, time_steps, _ = β.shape
    
    n = torch.arange(1, n_harmonic + 1, device=device, dtype=dtype).view(1, 1, -1)
    n_expanded = n.expand(batch_size, time_steps, -1)
    β_expanded = β.expand(-1, -1, n_harmonic)
    
    notch = torch.sin(torch.pi * n_expanded * β_expanded) ** 2
    return notch


def smooth_notch(notch: torch.Tensor, width: float, n_harmonic: int) -> torch.Tensor:
    """Applies frequency-domain smoothing to the bow notch."""
    device = notch.device
    dtype = notch.dtype
    batch_size, time_steps, _ = notch.shape
    
    n_fft = 2 * n_harmonic
    notch_padded = torch.nn.functional.pad(notch, (0, n_fft - n_harmonic))
    
    notch_fft = torch.fft.rfft(notch_padded, dim=-1)
    
    freq_bins = notch_fft.shape[-1]
    freqs = torch.linspace(0, 0.5, freq_bins, device=device, dtype=dtype)
    
    sigma = width / (2.0 * torch.pi)
    window = torch.exp(-0.5 * (freqs / sigma) ** 2)
    window = window.view(1, 1, -1).expand(batch_size, time_steps, -1)
    
    notch_fft_smooth = notch_fft * window
    notch_smooth_full = torch.fft.irfft(notch_fft_smooth, n=n_fft, dim=-1)
    notch_smooth = notch_smooth_full[:, :, :n_harmonic]
    
    return notch_smooth


def physics_harmonic_composer(β: torch.Tensor, γ: torch.Tensor, α: torch.Tensor,
                              residuals: torch.Tensor, B: torch.Tensor, pitch: torch.Tensor,
                              n_harmonic: int, notch_width: float, sampling_rate: float) -> tuple:
    """Composes harmonic magnitudes using violin physics model."""
    device = β.device
    dtype = β.dtype
    batch_size, time_steps, _ = β.shape
    n_residuals = residuals.shape[-1]
    
    n = torch.arange(1, n_harmonic + 1, device=device, dtype=dtype).view(1, 1, -1)
    n = n.expand(batch_size, time_steps, -1)
    
    baseline = 1.0 / n
    
    bow_notch = compute_bow_notch(β, n_harmonic)
    bow_notch_smooth = smooth_notch(bow_notch, notch_width, n_harmonic)
    bow_mask = 1.0 - γ * (1.0 - bow_notch_smooth)
    
    brightness_tilt = n ** (-α)
    
    spectrum = baseline * bow_mask * brightness_tilt
    
    residuals_full = torch.ones(batch_size, time_steps, n_harmonic, device=device, dtype=dtype)
    residuals_full[:, :, :n_residuals] = residuals
    spectrum = spectrum * residuals_full
    
    nyquist = sampling_rate / 2.0
    base_frequencies = pitch * n
    inharmonicity_factor = torch.sqrt(1.0 + B * (n ** 2))
    frequencies = base_frequencies * inharmonicity_factor
    nyquist_mask = (frequencies < nyquist).to(dtype)
    spectrum_culled = spectrum * nyquist_mask
    
    spectrum_culled_sum = spectrum_culled.sum(dim=-1, keepdim=True)
    safe_mask = (spectrum_culled_sum > 1e-7).to(dtype)
    fallback = torch.zeros_like(spectrum_culled)
    fallback[:, :, 0] = 1.0
    
    magnitudes_normalized = torch.where(
        safe_mask.expand_as(spectrum_culled) > 0.5,
        spectrum_culled / (spectrum_culled_sum + 1e-7),
        fallback
    )
    
    diagnostics = {
        'β': β,
        'γ': γ,
        'α': α,
        'B': B,
        'residuals': residuals,
        'bow_notch_smooth': bow_notch_smooth,
        'bow_mask': bow_mask,
        'brightness_tilt': brightness_tilt,
        'spectrum_before_culling': spectrum,
        'nyquist_mask': nyquist_mask,
        'spectrum_after_culling': spectrum_culled
    }
    
    return magnitudes_normalized, diagnostics