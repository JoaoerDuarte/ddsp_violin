import torch
import torch.nn as nn
from .nn import MLP, GRU
from .model_utils import (
    F0_SCALED, LD_SCALED, Z, AMPS, HARMONIC_DISTRIBUTION, NOISE_MAGNITUDES,
    BOW_POSITION_RAW, NOTCH_DEPTH_RAW, BRIGHTNESS_RAW, RESIDUALS_RAW,
    INHARMONICITY_COEFF
)


class Decoder(nn.Module):
    """Decodes conditioning inputs into synthesis parameters with dynamic output structure."""
    
    def __init__(self, inputs: list[str], z_dims: int | None = None, rnn_channels: int = 512,
                 ch: int = 512, layers_per_stack: int = 3, use_helmholtz_config: bool = False,
                 use_inharmonicity_config: bool = True, n_harmonic_config: int = 60,
                 n_bands_config: int = 65, n_residuals_config: int = 10,
                 use_bow_mask: bool = True, use_brightness_mask: bool = True,
                 use_residuals_mask: bool = True):
        super().__init__()

        self.input_names = inputs
        self.use_bow_mask = use_bow_mask
        self.use_brightness_mask = use_brightness_mask
        self.use_residuals_mask = use_residuals_mask

        if Z in self.input_names and z_dims is None:
            raise ValueError("z_dims must be provided if Z is in decoder inputs.")

        self.output_structure = self._build_output_structure(
            use_helmholtz_config, use_inharmonicity_config, n_harmonic_config,
            n_bands_config, n_residuals_config, use_bow_mask, use_brightness_mask,
            use_residuals_mask
        )

        self.input_stacks = nn.ModuleDict({
            name: MLP(z_dims if name == Z else 1, ch, layers_per_stack)
            for name in self.input_names
        })

        self.gru = GRU(len(self.input_names) * ch, rnn_channels)

        skip_size = rnn_channels + len(self.input_names) * ch
        self.out_stack = MLP(skip_size, ch, layers_per_stack)

        total_output_units = sum(size for _, size in self.output_structure)
        self.dense_out = nn.Linear(ch, total_output_units)

    def _build_output_structure(self, use_helmholtz, use_inharmonicity, n_harmonic, n_bands, 
                                n_residuals, use_bow_mask, use_brightness_mask, use_residuals_mask):
        """Build output structure based on configuration flags."""
        structure = [(AMPS, 1)]

        if use_helmholtz:
            if use_bow_mask:
                structure.append((BOW_POSITION_RAW, 1))
                structure.append((NOTCH_DEPTH_RAW, 1))
            if use_brightness_mask:
                structure.append((BRIGHTNESS_RAW, 1))
            if use_residuals_mask:
                structure.append((RESIDUALS_RAW, n_residuals))
        else:
            structure.append((HARMONIC_DISTRIBUTION, n_harmonic))
            if use_brightness_mask:
                structure.append((BRIGHTNESS_RAW, 1))

        if use_inharmonicity:
            structure.append((INHARMONICITY_COEFF, 1))

        structure.append((NOISE_MAGNITUDES, n_bands))
        return structure

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        for key in self.input_names:
            if key not in inputs:
                raise KeyError(f"Missing required decoder input: {key}")

        processed_inputs = []
        for name in self.input_names:
            x = inputs[name]
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            elif x.dim() != 3:
                raise ValueError(f"Decoder input '{name}' has unexpected shape: {x.shape}. Expected [batch, time, ch].")
            processed_inputs.append(self.input_stacks[name](x))

        concatenated = torch.cat(processed_inputs, dim=-1)
        gru_output = self.gru(concatenated)[0]

        skip_connected = torch.cat([gru_output] + processed_inputs, dim=-1)
        processed = self.out_stack(skip_connected)
        raw_output = self.dense_out(processed)

        results = {}
        current_idx = 0
        for name, size in self.output_structure:
            results[name] = raw_output[..., current_idx:current_idx + size]
            current_idx += size

        return results