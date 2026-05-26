import torch
import torch.nn as nn
from typing import Dict

from .core import exp_sigmoid, remove_above_nyquist, scale_f0_hz, scale_db
from .synth import harmonic_synth, filtered_noise_synth
from .encoder import Encoder
from .decoder import Decoder
from .model_utils import (
    F0_SCALED, LD_SCALED, Z, AMPS, HARMONIC_DISTRIBUTION, NOISE_MAGNITUDES,
    BOW_POSITION_RAW, NOTCH_DEPTH_RAW, BRIGHTNESS_RAW, RESIDUALS_RAW,
    create_filter, print_model_summary,
    activate_bow_position, activate_notch_depth, activate_brightness, activate_residuals,
    physics_harmonic_composer,
)


class DDSP(nn.Module):
    """Dual-mode DDSP model supporting standard and violin (physics-guided) synthesis."""

    def __init__(self, **config):
        super().__init__()
        self.model_config = config
        self.encoder_config = config.get("encoder", {})
        self.decoder_config = config["decoder"]

        self.resonance_position = config.get("resonance_position", "before_noise")
        if self.resonance_position not in ["before_noise", "after_noise"]:
            raise ValueError(f"resonance_position must be 'before_noise' or 'after_noise', got {self.resonance_position}")

        helmholtz_config = config.get("helmholtz", {})
        self.use_helmholtz = bool(helmholtz_config.get("use_helmholtz_synthesis", False))

        if self.use_helmholtz:
            self.β_min = float(helmholtz_config.get("β_min", 0.05))
            self.β_max = float(helmholtz_config.get("β_max", 0.50))
            self.α_min = float(helmholtz_config.get("α_min", -1.0))
            self.α_max = float(helmholtz_config.get("α_max", 1.0))
            self.n_residuals = int(helmholtz_config.get("n_residuals", 10))
            self.residual_scale = float(helmholtz_config.get("residual_scale", 1.0))

            self.use_bow_mask = bool(helmholtz_config.get("use_bow_mask", True))
            self.use_brightness_mask = bool(helmholtz_config.get("use_brightness_mask", True))
            self.use_residuals_mask = bool(helmholtz_config.get("use_residuals_mask", True))

            self.violin_config = {
                'β_min': self.β_min, 'β_max': self.β_max,
                'α_min': self.α_min, 'α_max': self.α_max,
                'n_residuals': self.n_residuals, 'residual_scale': self.residual_scale,
                'use_bow_mask': self.use_bow_mask,
                'use_brightness_mask': self.use_brightness_mask,
                'use_residuals_mask': self.use_residuals_mask,
            }
            self.use_brightness_tilt_standard = False
            self.brightness_config_standard = {}
        else:
            self.use_bow_mask = True
            self.use_brightness_mask = True
            self.use_residuals_mask = True
            self.violin_config = {}

            self.use_brightness_tilt_standard = bool(config.get("use_brightness_tilt_standard", False))
            if self.use_brightness_tilt_standard:
                self.α_min_standard = float(config.get("standard_brightness_α_min",
                                                       helmholtz_config.get("α_min", -1.0)))
                self.α_max_standard = float(config.get("standard_brightness_α_max",
                                                       helmholtz_config.get("α_max", 1.0)))
                self.brightness_config_standard = {'α_min': self.α_min_standard, 'α_max': self.α_max_standard}
            else:
                self.brightness_config_standard = {}

        self.register_buffer("sampling_rate", torch.tensor(float(config["sampling_rate"])))
        self.register_buffer("block_size", torch.tensor(int(config["block_size"])))
        self.register_buffer("noise_bias", torch.tensor(float(self.decoder_config.get("noise_bias", -5.0))))

        self.n_harmonic = int(self.decoder_config["n_harmonic"])
        n_bands = int(self.decoder_config["n_bands"])

        self._initialize_encoder()
        self._initialize_decoder(n_bands)
        self._initialize_filters()

        print_model_summary(
            self.encoder, self.decoder, self.resonance, self.room,
            self.use_helmholtz, self.violin_config,
            use_brightness_standard=self.use_brightness_tilt_standard,
            brightness_config_standard=self.brightness_config_standard,
            decoder_output_structure=self.decoder.output_structure,
            resonance_position=self.resonance_position if self.resonance is not None else None,
        )

    def _initialize_encoder(self):
        """Build the MFCC + GRU audio encoder if enabled."""
        if not bool(self.encoder_config.get("use_encoder", False)):
            self.encoder = None
            return

        signal_length = int(self.model_config.get("signal_length", 64000))
        block_size_val = int(self.block_size.item())
        if block_size_val <= 0:
            raise ValueError("block_size must be positive")

        self.encoder = Encoder(
            rnn_channels=int(self.encoder_config["rnn_channels"]),
            z_dims=int(self.encoder_config["z_dims"]),
            z_time_steps=int(self.encoder_config["z_time_steps"]),
            sample_rate=int(self.sampling_rate.item()),
            target_length=signal_length // block_size_val,
            mfcc_config=self.encoder_config.get("mfcc"),
        )

    def _initialize_decoder(self, n_bands):
        """Build the conditioning → synthesis-parameters decoder."""
        n_residuals = self.n_residuals if self.use_helmholtz else 10

        decoder_inputs = [F0_SCALED, LD_SCALED]
        if self.encoder is not None:
            decoder_inputs.append(Z)

        decoder_use_brightness = self.use_brightness_mask if self.use_helmholtz else self.use_brightness_tilt_standard

        self.decoder = Decoder(
            inputs=decoder_inputs,
            z_dims=int(self.encoder_config.get("z_dims")) if self.encoder and self.encoder_config.get("z_dims") else None,
            rnn_channels=int(self.decoder_config["rnn_channels"]),
            ch=int(self.decoder_config["ch"]),
            layers_per_stack=int(self.decoder_config["layers_per_stack"]),
            use_helmholtz_config=self.use_helmholtz,
            n_harmonic_config=self.n_harmonic,
            n_bands_config=n_bands,
            n_residuals_config=n_residuals,
            use_bow_mask=self.use_bow_mask,
            use_brightness_mask=decoder_use_brightness,
            use_residuals_mask=self.use_residuals_mask,
            init_harmonic_1_over_n=bool(self.decoder_config.get("init_harmonic_1_over_n", False)),
        )

    def _initialize_filters(self):
        """Build resonance and (optional) room filters."""
        self.resonance = self._create_filter("resonance", self.model_config.get("use_resonance", False))
        self.room = self._create_filter("room", self.model_config.get("use_room", False))

    def _create_filter(self, filter_name, use_filter):
        if not use_filter:
            return None
        filter_type = str(self.model_config.get(f"{filter_name}_type", "ar"))
        return create_filter(filter_name, filter_type, self.model_config)

    def forward(self, pitch: torch.Tensor, loudness: torch.Tensor,
                audio: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        if pitch.dim() == 2:
            pitch = pitch.unsqueeze(-1)
        if loudness.dim() == 2:
            loudness = loudness.unsqueeze(-1)

        decoder_inputs = {F0_SCALED: scale_f0_hz(pitch), LD_SCALED: scale_db(loudness)}

        if self.encoder is not None:
            if audio is None:
                raise ValueError("Audio input required for encoder")
            if audio.dim() == 3 and audio.shape[-1] == 1:
                audio = audio.squeeze(-1)
            elif audio.dim() != 2:
                raise ValueError(f"Expected audio [B, L], got {audio.shape}")
            decoder_inputs[Z] = self.encoder(audio)

        synthesis_params = self.decoder(**decoder_inputs)

        total_amplitude = exp_sigmoid(synthesis_params[AMPS])
        noise_magnitudes = exp_sigmoid(synthesis_params[NOISE_MAGNITUDES] + self.noise_bias)

        harmonic_signal, output_data = self._generate_harmonic_component(
            synthesis_params, pitch, total_amplitude,
        )

        if self.resonance is not None and self.resonance_position == "before_noise":
            harmonic_signal = self.resonance(harmonic_signal)

        noise_component = filtered_noise_synth(noise_magnitudes, int(self.block_size.item()))
        combined_signal = harmonic_signal + noise_component

        if self.resonance is not None and self.resonance_position == "after_noise":
            combined_signal = self.resonance(combined_signal)

        final_signal = self.room(combined_signal) if self.room is not None else combined_signal

        if final_signal.dim() == 2:
            final_signal = final_signal.unsqueeze(-1)

        output_data['signal'] = final_signal
        return output_data

    def _generate_harmonic_component(self, synthesis_params, pitch, total_amplitude):
        """Dispatch to violin (physics-guided) or standard DDSP harmonic synthesis."""
        sampling_rate = int(self.sampling_rate.item())
        block_size = int(self.block_size.item())

        if self.use_helmholtz:
            return self._generate_violin_harmonic(synthesis_params, pitch, total_amplitude, sampling_rate, block_size)
        return self._generate_standard_harmonic(synthesis_params, pitch, total_amplitude, sampling_rate, block_size)

    def _generate_violin_harmonic(self, synthesis_params, pitch, total_amplitude, sampling_rate, block_size):
        """Synthesize harmonic component from (β, γ, α, ρ_n) physics parameters."""
        batch_size, time_steps = pitch.shape[0], pitch.shape[1]
        device, dtype = pitch.device, pitch.dtype

        if BOW_POSITION_RAW in synthesis_params:
            β = activate_bow_position(synthesis_params[BOW_POSITION_RAW], self.β_min, self.β_max)
            γ = activate_notch_depth(synthesis_params[NOTCH_DEPTH_RAW])
        else:
            β = torch.zeros(batch_size, time_steps, 1, device=device, dtype=dtype)
            γ = torch.zeros(batch_size, time_steps, 1, device=device, dtype=dtype)

        if BRIGHTNESS_RAW in synthesis_params:
            α = activate_brightness(synthesis_params[BRIGHTNESS_RAW], self.α_min, self.α_max)
        else:
            α = torch.zeros(batch_size, time_steps, 1, device=device, dtype=dtype)

        if RESIDUALS_RAW in synthesis_params:
            residuals = activate_residuals(synthesis_params[RESIDUALS_RAW], self.residual_scale)
        else:
            residuals = torch.ones(batch_size, time_steps, self.n_residuals, device=device, dtype=dtype)

        magnitudes_normalized, diagnostics = physics_harmonic_composer(
            β=β, γ=γ, α=α, residuals=residuals, pitch=pitch,
            n_harmonic=self.n_harmonic,
            sampling_rate=float(sampling_rate),
            use_bow_mask=self.use_bow_mask,
            use_brightness_mask=self.use_brightness_mask,
            use_residuals_mask=self.use_residuals_mask,
        )

        harmonic_signal = harmonic_synth(
            pitch_frames=pitch,
            amplitudes_dist_normalized_frames=magnitudes_normalized,
            total_amp_frames=total_amplitude,
            sampling_rate=sampling_rate,
            block_size=block_size,
        )
        return harmonic_signal, {'violin_diagnostics': diagnostics}

    def _generate_standard_harmonic(self, synthesis_params, pitch, total_amplitude, sampling_rate, block_size):
        """Synthesize harmonic component from unconstrained per-harmonic amplitudes (standard DDSP)."""
        if HARMONIC_DISTRIBUTION not in synthesis_params:
            raise RuntimeError(f"'{HARMONIC_DISTRIBUTION}' expected from Decoder.")

        harmonic_amplitudes = exp_sigmoid(synthesis_params[HARMONIC_DISTRIBUTION])

        if self.use_brightness_tilt_standard and BRIGHTNESS_RAW in synthesis_params:
            α = activate_brightness(synthesis_params[BRIGHTNESS_RAW], self.α_min_standard, self.α_max_standard)
            device, dtype = harmonic_amplitudes.device, harmonic_amplitudes.dtype
            n = torch.arange(1, self.n_harmonic + 1, device=device, dtype=dtype).view(1, 1, -1)
            harmonic_amplitudes = harmonic_amplitudes * (n ** (-α))

        harmonic_amplitudes = remove_above_nyquist(harmonic_amplitudes, pitch, float(sampling_rate))

        amplitude_sum = harmonic_amplitudes.sum(-1, keepdim=True)
        normalized_amplitudes = harmonic_amplitudes / (amplitude_sum + 1e-7)

        harmonic_signal = harmonic_synth(
            pitch_frames=pitch,
            amplitudes_dist_normalized_frames=normalized_amplitudes,
            total_amp_frames=total_amplitude,
            sampling_rate=sampling_rate,
            block_size=block_size,
        )
        return harmonic_signal, {'harmonic_amplitudes': normalized_amplitudes}
