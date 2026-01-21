import torch
import torch.nn as nn
import torchaudio
from .nn import MLP, GRU, Normalize


class Encoder(nn.Module):
    """Encodes audio into a latent representation using MFCCs and a GRU."""
    
    def __init__(self, rnn_channels: int = 512, z_dims: int = 16, z_time_steps: int = 125,
                 sample_rate: int = 16000, target_length: int = 1000, mfcc_config: dict = None):
        super().__init__()

        self.target_length = target_length

        # Default MFCC settings
        if mfcc_config is None:
            mfcc_config = {
                'n_mfcc': 30, 'n_mels': 128, 'f_min': 20.0,
                'f_max': 8000.0, 'log_mels': True
            }

        # Determine FFT/hop sizes based on z_time_steps
        self.fft_size, self.hop_length = self._get_stft_params(z_time_steps)

        # MFCC extraction layer
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=mfcc_config.get('n_mfcc', 30),
            log_mels=mfcc_config.get('log_mels', True),
            melkwargs={
                'n_fft': self.fft_size,
                'n_mels': mfcc_config.get('n_mels', 128),
                'hop_length': self.hop_length,
                'f_min': mfcc_config.get('f_min', 20.0),
                'f_max': mfcc_config.get('f_max', 8000.0),
            }
        )

        # Processing layers
        n_mfcc = mfcc_config.get('n_mfcc', 30)
        self.norm = Normalize(n_mfcc)
        self.gru = GRU(n_mfcc, rnn_channels, multiply_input_size=False)
        self.dense_out = nn.Linear(rnn_channels, z_dims)

    def _get_stft_params(self, z_time_steps):
        """Get FFT size and hop length based on desired time steps."""
        valid_steps = {
            63: {'fft_size': 2048, 'overlap': 0.5},
            125: {'fft_size': 1024, 'overlap': 0.5},
            250: {'fft_size': 1024, 'overlap': 0.75},
            500: {'fft_size': 512, 'overlap': 0.75},
            1000: {'fft_size': 256, 'overlap': 0.75}
        }
        
        if z_time_steps not in valid_steps:
            raise ValueError(f'z_time_steps must be one of {list(valid_steps.keys())}, got {z_time_steps}')
        
        spec = valid_steps[z_time_steps]
        fft_size = spec['fft_size']
        hop_length = int(fft_size * (1 - spec['overlap']))
        
        return fft_size, hop_length

    def compute_z(self, audio):
        """Computes latent vectors before temporal interpolation."""
        # Extract MFCCs: [batch, n_mfcc, time_mfcc]
        z_mfcc = self.mfcc_transform(audio)
        z_mfcc = z_mfcc.transpose(1, 2)  # [batch, time_mfcc, n_mfcc]

        # Process through normalization, GRU, and projection
        z_normalized = self.norm(z_mfcc)
        z_gru_output, _ = self.gru(z_normalized)
        z_projected = self.dense_out(z_gru_output)
        
        return z_projected

    def expand_z(self, z, target_time_steps):
        """Interpolates the time dimension of z to match target_time_steps."""
        if z.shape[1] == target_time_steps:
            return z

        # Interpolate time dimension: [batch, time, z_dims] -> [batch, target_time, z_dims]
        z_transposed = z.transpose(1, 2)  # [batch, z_dims, time]
        z_interpolated = torch.nn.functional.interpolate(
            z_transposed,
            size=target_time_steps,
            mode='linear',
            align_corners=False
        )
        return z_interpolated.transpose(1, 2)  # [batch, target_time, z_dims]

    def forward(self, audio):
        """Runs the encoder: MFCC -> Normalize -> GRU -> Linear -> Interpolate."""
        z_raw = self.compute_z(audio)
        z_expanded = self.expand_z(z_raw, self.target_length)
        return z_expanded