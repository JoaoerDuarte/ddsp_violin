import torch
from typing import Dict, Any

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


def count_parameters(model):
    """Trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_filter(filter_name: str, filter_type: str, config: Dict[str, Any]):
    """Build a resonance/room filter of the requested type ('convolutional', 'ar', 'arma')."""
    from .filters import ConvolutionalFilter, ARFilter, ARMAFilter

    is_room = filter_name == "room"

    try:
        if filter_type == "ar":
            return ARFilter(**extract_filter_config(config, is_room=is_room))
        elif filter_type == "arma":
            return ARMAFilter(**extract_filter_config(config, is_room=is_room))
        elif filter_type == "convolutional":
            return ConvolutionalFilter(**extract_conv_config(config, is_resonance=not is_room))
        else:
            print(f"Warning: Unknown {filter_name} filter type '{filter_type}', skipping.")
            return None
    except Exception as e:
        print(f"Warning: Failed to initialize {filter_name} filter '{filter_type}': {e}")
        return None


def extract_conv_config(config: Dict[str, Any], is_resonance: bool = True) -> Dict[str, Any]:
    """Read FIR filter kwargs from the model config block."""
    prefix = "resonance_" if is_resonance else "room_"
    default_add_dry = False if is_resonance else True
    default_mask_ir = False if is_resonance else True
    default_normalization = "psd_weighted_rms"

    normalization_type = config.get(f"{prefix}normalization_type", default_normalization)
    conv_config = {
        "length": int(config.get(f"{prefix}length", 48000)),
        "add_dry": bool(config.get(f"{prefix}add_dry", default_add_dry)),
        "mask_ir": bool(config.get(f"{prefix}mask_ir", default_mask_ir)),
        "normalization_type": normalization_type,
    }
    if normalization_type == "psd_weighted_rms":
        conv_config["n_fft"] = int(config.get(f"{prefix}n_fft", 2048))
        conv_config["psd_prior"] = str(config.get(f"{prefix}psd_prior", "pink"))
        conv_config["psd_ema_alpha"] = float(config.get(f"{prefix}psd_ema_alpha", 0.99))
    return conv_config


def extract_filter_config(config: Dict[str, Any], is_room: bool = False) -> Dict[str, Any]:
    """Read AR/ARMA filter kwargs from the model config block."""
    prefix = "room_" if is_room else "resonance_"
    default_add_dry = True if is_room else False

    filter_config = {
        "add_dry": bool(config.get(f"{prefix}add_dry", default_add_dry)),
        "max_reflection": float(config.get(f"{prefix}max_reflection", 0.999)),
        "normalization_type": config.get(f"{prefix}normalization_type", "rms"),
    }

    f_type = config.get(f"{prefix}type") or config.get("room_type" if is_room else "resonance_type")
    if f_type == "ar":
        filter_config["lpc_order"] = int(config.get(f"{prefix}ar_order", 16))
        filter_config["window_size"] = int(config.get(f"{prefix}window_size", 4096))
    elif f_type == "arma":
        filter_config["ar_order"] = int(config.get(f"{prefix}ar_order", 16))
        filter_config["ma_order"] = int(config.get(f"{prefix}ma_order", 16))
        filter_config["window_size"] = int(config.get(f"{prefix}window_size", 4096))

    return filter_config


def print_model_summary(encoder, decoder, resonance, room, use_helmholtz, violin_config,
                        use_brightness_standard=False, brightness_config_standard=None,
                        decoder_output_structure=None, resonance_position=None):
    """Print a formatted summary of model architecture and synthesis configuration."""
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)

    encoder_params = count_parameters(encoder) if encoder is not None else 0
    print(f"\nEncoder: {encoder_params:,} parameters" if encoder is not None else "\nEncoder: Not used")

    decoder_params = count_parameters(decoder)
    print(f"Decoder: {decoder_params:,} parameters")

    if decoder_output_structure is not None:
        print("\nDecoder Output Structure:")
        total_units = 0
        for output_name, output_size in decoder_output_structure:
            annotation = ""
            if output_name == BRIGHTNESS_RAW:
                annotation = " (Violin Brightness)" if use_helmholtz else (" (Standard Brightness Tilt)" if use_brightness_standard else "")
            elif output_name == HARMONIC_DISTRIBUTION:
                annotation = " (Standard Mode)"
            elif output_name == BOW_POSITION_RAW:
                annotation = " (Violin Bow Position β)"
            elif output_name == NOTCH_DEPTH_RAW:
                annotation = " (Violin Notch Depth γ)"
            elif output_name == RESIDUALS_RAW:
                annotation = " (Violin Residuals)"
            print(f"  - {output_name}: {output_size} unit{'s' if output_size > 1 else ''}{annotation}")
            total_units += output_size
        print(f"  Total Decoder Outputs: {total_units} units")

    print()
    resonance_params = 0
    if resonance is not None:
        resonance_params = count_parameters(resonance)
        print(f"Resonance Filter ({resonance.__class__.__name__}): {resonance_params:,} parameters")
        if resonance_position is not None:
            print(f"  - Position: {resonance_position}")
    else:
        print("Resonance Filter: Not used")

    room_params = 0
    if room is not None:
        room_params = count_parameters(room)
        print(f"Room Filter ({room.__class__.__name__}): {room_params:,} parameters")
    else:
        print("Room Filter: Not used")

    total_params = decoder_params + encoder_params + resonance_params + room_params
    print(f"\nTotal Parameters: {total_params:,}")

    print("\n" + "-" * 60)
    print("SYNTHESIS CONFIGURATION")
    print("-" * 60)

    if use_helmholtz:
        print("\nSynthesis Mode: Violin (Physics-Guided)")
        use_bow = violin_config.get('use_bow_mask', True)
        use_brightness = violin_config.get('use_brightness_mask', True)
        use_residuals = violin_config.get('use_residuals_mask', True)

        print(f"  - Bow Mask (β, γ): {'Enabled' if use_bow else 'Disabled'}")
        if use_bow:
            print(f"    - Bow Position Range: [{violin_config.get('β_min', 'N/A')}, {violin_config.get('β_max', 'N/A')}]")

        print(f"  - Brightness Mask (α): {'Enabled' if use_brightness else 'Disabled'}")
        if use_brightness:
            print(f"    - Brightness Range: [{violin_config.get('α_min', 'N/A')}, {violin_config.get('α_max', 'N/A')}]")

        print(f"  - Residuals Mask: {'Enabled' if use_residuals else 'Disabled'}")
        if use_residuals:
            print(f"    - Residuals: {violin_config.get('n_residuals', 'N/A')} (scale: {violin_config.get('residual_scale', 'N/A')})")
    else:
        print("\nSynthesis Mode: Standard DDSP")
        print(f"  - Brightness Tilt: {'Enabled' if use_brightness_standard else 'Disabled'}")
        if use_brightness_standard and brightness_config_standard is not None:
            print(f"    - α range: [{brightness_config_standard.get('α_min', 'N/A')}, {brightness_config_standard.get('α_max', 'N/A')}]")

    print("=" * 60 + "\n")


def activate_bow_position(raw: torch.Tensor, β_min: float, β_max: float) -> torch.Tensor:
    """Scaled sigmoid: maps raw → β in [β_min, β_max]."""
    return β_min + (β_max - β_min) * torch.sigmoid(raw)


def activate_notch_depth(raw: torch.Tensor) -> torch.Tensor:
    """Maps raw → γ in [0, 1] via sigmoid."""
    return torch.sigmoid(raw)


def activate_brightness(raw: torch.Tensor, α_min: float, α_max: float) -> torch.Tensor:
    """Scaled sigmoid: maps raw → α in [α_min, α_max]."""
    return α_min + (α_max - α_min) * torch.sigmoid(raw)


def activate_residuals(raw: torch.Tensor, residual_scale: float) -> torch.Tensor:
    """Maps raw → ρ_n = 1 + scale·tanh(raw), centered at unity."""
    return 1.0 + residual_scale * torch.tanh(raw)


def compute_bow_notch(β: torch.Tensor, n_harmonic: int) -> torch.Tensor:
    """Helmholtz bow notch: sin²(πnβ) per harmonic."""
    device, dtype = β.device, β.dtype
    batch_size, time_steps, _ = β.shape
    n = torch.arange(1, n_harmonic + 1, device=device, dtype=dtype).view(1, 1, -1).expand(batch_size, time_steps, -1)
    β_expanded = β.expand(-1, -1, n_harmonic)
    return torch.sin(torch.pi * n * β_expanded) ** 2


def physics_harmonic_composer(β: torch.Tensor, γ: torch.Tensor, α: torch.Tensor,
                              residuals: torch.Tensor, pitch: torch.Tensor,
                              n_harmonic: int, sampling_rate: float,
                              use_bow_mask: bool = True, use_brightness_mask: bool = True,
                              use_residuals_mask: bool = True) -> tuple:
    """Compose normalized harmonic magnitudes from the (β, γ, α, ρ_n) physics parameters."""
    device, dtype = β.device, β.dtype
    batch_size, time_steps, _ = β.shape
    n_residuals = residuals.shape[-1]

    n = torch.arange(1, n_harmonic + 1, device=device, dtype=dtype).view(1, 1, -1).expand(batch_size, time_steps, -1)
    baseline = 1.0 / n

    if use_bow_mask:
        bow_notch = compute_bow_notch(β, n_harmonic)
        bow_mask = 1.0 - γ * (1.0 - bow_notch)
    else:
        bow_mask = torch.ones_like(baseline)

    brightness_tilt = n ** (-α) if use_brightness_mask else torch.ones_like(baseline)
    spectrum = baseline * bow_mask * brightness_tilt

    if use_residuals_mask:
        residuals_full = torch.ones(batch_size, time_steps, n_harmonic, device=device, dtype=dtype)
        residuals_full[:, :, :n_residuals] = residuals
        spectrum = spectrum * residuals_full

    nyquist = sampling_rate / 2.0
    nyquist_mask = (pitch * n < nyquist).to(dtype)
    spectrum_culled = spectrum * nyquist_mask

    spectrum_sum = spectrum_culled.sum(dim=-1, keepdim=True)
    safe = (spectrum_sum > 1e-7).to(dtype)
    fallback = torch.zeros_like(spectrum_culled)
    fallback[:, :, 0] = 1.0
    magnitudes_normalized = torch.where(
        safe.expand_as(spectrum_culled) > 0.5,
        spectrum_culled / (spectrum_sum + 1e-7),
        fallback,
    )

    diagnostics = {
        'β': β,
        'γ': γ,
        'α': α,
        'residuals': residuals,
        'bow_mask': bow_mask,
        'brightness_tilt': brightness_tilt,
        'spectrum_before_culling': spectrum,
        'nyquist_mask': nyquist_mask,
        'spectrum_after_culling': spectrum_culled,
    }
    return magnitudes_normalized, diagnostics
