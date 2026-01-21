import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .core import exp_sigmoid, remove_above_nyquist, scale_f0_hz, scale_db
from .synth import harmonic_synth, filtered_noise_synth
from .encoder import Encoder
from .decoder import Decoder
from .model_utils import (
    F0_SCALED, LD_SCALED, Z, AMPS, HARMONIC_DISTRIBUTION, NOISE_MAGNITUDES,
    BOW_POSITION_RAW, NOTCH_DEPTH_RAW, BRIGHTNESS_RAW, RESIDUALS_RAW,
    INHARMONICITY_COEFF,
    count_parameters, create_filter, get_inharmonicity_coefficient,
    print_model_summary, activate_bow_position, activate_notch_depth,
    activate_brightness, activate_residuals, physics_harmonic_composer
)


class DDSP(nn.Module):
    """Dual-mode DDSP model supporting standard and violin (physics-guided) synthesis."""

    def __init__(self, **config):
        super().__init__()
        self.model_config = config
        self.encoder_config = config.get("encoder", {})
        self.decoder_config = config["decoder"]

        self.use_inharmonicity = bool(config.get("use_inharmonicity", True))
        
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
            self.notch_width = float(helmholtz_config.get("notch_width", 0.05))
            self.n_residuals = int(helmholtz_config.get("n_residuals", 10))
            self.residual_scale = float(helmholtz_config.get("residual_scale", 1.0))
            self.inharmonicity_b_max = float(helmholtz_config.get("inharmonicity_b_max", 0.0005))
            self.violin_config = {
                'β_min': self.β_min,
                'β_max': self.β_max,
                'α_min': self.α_min,
                'α_max': self.α_max,
                'notch_width': self.notch_width,
                'n_residuals': self.n_residuals,
                'residual_scale': self.residual_scale
            }
        else:
            self.violin_config = {}

        self.register_buffer("sampling_rate", torch.tensor(float(config["sampling_rate"])))
        self.register_buffer("block_size", torch.tensor(int(config["block_size"])))
        self.register_buffer("noise_bias", torch.tensor(float(self.decoder_config.get("noise_bias", -5.0))))

        self.n_harmonic = int(self.decoder_config["n_harmonic"])
        n_bands = int(self.decoder_config["n_bands"])

        self._initialize_encoder()
        self._initialize_decoder(n_bands)
        self._initialize_filters()
        
        if hasattr(self, 'resonance') and self.resonance is not None:
            print(f"- Resonance Position: {self.resonance_position}")
        
        print_model_summary(
            self.encoder, self.decoder, self.resonance, self.room,
            self.use_helmholtz, self.violin_config, self.use_inharmonicity
        )

    def _initialize_encoder(self):
        """Initialize encoder if configured."""
        use_encoder = bool(self.encoder_config.get("use_encoder", False))
        
        if use_encoder:
            signal_length = int(self.model_config.get("signal_length", 64000))
            block_size_val = int(self.block_size.item())
            if block_size_val <= 0:
                raise ValueError("block_size must be positive")
            
            target_length = signal_length // block_size_val
            self.encoder = Encoder(
                rnn_channels=int(self.encoder_config["rnn_channels"]),
                z_dims=int(self.encoder_config["z_dims"]),
                z_time_steps=int(self.encoder_config["z_time_steps"]),
                sample_rate=int(self.sampling_rate.item()),
                target_length=target_length,
                mfcc_config=self.encoder_config.get("mfcc")
            )
        else:
            self.encoder = None

    def _initialize_decoder(self, n_bands):
        """Initialize decoder with dynamic output structure."""
        n_residuals = self.n_residuals if self.use_helmholtz else 10

        decoder_inputs = [F0_SCALED, LD_SCALED]
        if self.encoder is not None:
            decoder_inputs.append(Z)

        self.decoder = Decoder(
            inputs=decoder_inputs,
            z_dims=int(self.encoder_config.get("z_dims")) if self.encoder and self.encoder_config.get("z_dims") else None,
            rnn_channels=int(self.decoder_config["rnn_channels"]),
            ch=int(self.decoder_config["ch"]),
            layers_per_stack=int(self.decoder_config["layers_per_stack"]),
            use_helmholtz_config=self.use_helmholtz,
            use_inharmonicity_config=self.use_inharmonicity,
            n_harmonic_config=self.n_harmonic,
            n_bands_config=n_bands,
            n_residuals_config=n_residuals
        )

    def _initialize_filters(self):
        """Initialize resonance and room filters if configured."""
        self.resonance = self._create_filter("resonance", self.model_config.get("use_resonance", False))
        self.room = self._create_filter("room", self.model_config.get("use_room", False))

    def _create_filter(self, filter_name, use_filter):
        """Create a filter of the specified type."""
        if not use_filter:
            return None

        filter_type = str(self.model_config.get(f"{filter_name}_type", "ar"))
        return create_filter(filter_name, filter_type, self.model_config)

    def forward(self, pitch: torch.Tensor, loudness: torch.Tensor, audio: torch.Tensor | None = None) -> Dict[str, torch.Tensor | None]:
        if pitch.dim() == 2:
            pitch = pitch.unsqueeze(-1)
        if loudness.dim() == 2:
            loudness = loudness.unsqueeze(-1)

        pitch_scaled = scale_f0_hz(pitch)
        loudness_scaled = scale_db(loudness)
        decoder_inputs = {F0_SCALED: pitch_scaled, LD_SCALED: loudness_scaled}

        if self.encoder is not None:
            if audio is None:
                raise ValueError("Audio input required for encoder")
            if audio.dim() == 3 and audio.shape[-1] == 1:
                audio = audio.squeeze(-1)
            elif audio.dim() != 2:
                raise ValueError(f"Expected audio [B, L], got {audio.shape}")
            
            z_latent = self.encoder(audio)
            decoder_inputs[Z] = z_latent

        synthesis_params = self.decoder(**decoder_inputs)

        total_amplitude = exp_sigmoid(synthesis_params[AMPS])
        noise_magnitudes = exp_sigmoid(synthesis_params[NOISE_MAGNITUDES] + self.noise_bias)

        inharmonicity_coeff = get_inharmonicity_coefficient(
            synthesis_params, pitch, self.use_inharmonicity, 
            self.inharmonicity_b_max if self.use_helmholtz else 0.0005
        )

        harmonic_signal, output_data = self._generate_harmonic_component(
            synthesis_params, pitch, total_amplitude, inharmonicity_coeff
        )

        if self.resonance is not None and self.resonance_position == "before_noise":
            harmonic_signal = self.resonance(harmonic_signal)

        noise_component = filtered_noise_synth(noise_magnitudes, int(self.block_size.item()))
        
        combined_signal = harmonic_signal + noise_component

        if self.resonance is not None and self.resonance_position == "after_noise":
            combined_signal = self.resonance(combined_signal)

        if self.room is not None:
            final_signal = self.room(combined_signal)
        else:
            final_signal = combined_signal

        if final_signal.dim() == 2:
            final_signal = final_signal.unsqueeze(-1)

        output_data['signal'] = final_signal
        return output_data

    def _generate_harmonic_component(self, synthesis_params, pitch, total_amplitude, inharmonicity_coeff):
        """Generate harmonic component based on synthesis mode."""
        sampling_rate = int(self.sampling_rate.item())
        block_size = int(self.block_size.item())

        if self.use_helmholtz:
            return self._generate_violin_harmonic(
                synthesis_params, pitch, total_amplitude, inharmonicity_coeff, sampling_rate, block_size
            )
        else:
            return self._generate_standard_harmonic(
                synthesis_params, pitch, total_amplitude, inharmonicity_coeff, sampling_rate, block_size
            )

    def _generate_violin_harmonic(self, synthesis_params, pitch, total_amplitude, inharmonicity_coeff, sampling_rate, block_size):
        """Generate harmonic component using violin (physics-guided) synthesis."""
        if BOW_POSITION_RAW not in synthesis_params:
            raise RuntimeError(f"'{BOW_POSITION_RAW}' expected from Decoder.")

        β = activate_bow_position(synthesis_params[BOW_POSITION_RAW], self.β_min, self.β_max)
        γ = activate_notch_depth(synthesis_params[NOTCH_DEPTH_RAW])
        α = activate_brightness(synthesis_params[BRIGHTNESS_RAW], self.α_min, self.α_max)
        residuals = activate_residuals(synthesis_params[RESIDUALS_RAW], self.residual_scale)

        magnitudes_normalized, diagnostics = physics_harmonic_composer(
            β=β,
            γ=γ,
            α=α,
            residuals=residuals,
            B=inharmonicity_coeff,
            pitch=pitch,
            n_harmonic=self.n_harmonic,
            notch_width=self.notch_width,
            sampling_rate=float(sampling_rate)
        )

        harmonic_signal = harmonic_synth(
            pitch_frames=pitch,
            amplitudes_dist_normalized_frames=magnitudes_normalized,
            total_amp_frames=total_amplitude,
            sampling_rate=sampling_rate,
            block_size=block_size,
            inharmonicity_coeff_frames=inharmonicity_coeff
        )

        return harmonic_signal, {'violin_diagnostics': diagnostics}

    def _generate_standard_harmonic(self, synthesis_params, pitch, total_amplitude, inharmonicity_coeff, sampling_rate, block_size):
        """Generate harmonic component using standard DDSP synthesis."""
        if HARMONIC_DISTRIBUTION not in synthesis_params:
            raise RuntimeError(f"'{HARMONIC_DISTRIBUTION}' expected from Decoder.")

        harmonic_amplitudes = exp_sigmoid(synthesis_params[HARMONIC_DISTRIBUTION])
        harmonic_amplitudes = remove_above_nyquist(
            harmonic_amplitudes, pitch, float(sampling_rate), inharmonicity_coeff
        )
        
        amplitude_sum = harmonic_amplitudes.sum(-1, keepdim=True)
        normalized_amplitudes = harmonic_amplitudes / (amplitude_sum + 1e-7)

        harmonic_signal = harmonic_synth(
            pitch_frames=pitch,
            amplitudes_dist_normalized_frames=normalized_amplitudes,
            total_amp_frames=total_amplitude,
            sampling_rate=sampling_rate,
            block_size=block_size,
            inharmonicity_coeff_frames=inharmonicity_coeff
        )

        return harmonic_signal, {'harmonic_amplitudes': normalized_amplitudes}