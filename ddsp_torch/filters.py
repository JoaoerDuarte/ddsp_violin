import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import lfilter
from .dsp import fft_convolve, frequency_to_impulse_response
from .filter_utils import reflection_to_ar


class ConvolutionalFilter(nn.Module):
    """Applies a learned finite impulse response filter using FFT convolution."""
    
    def __init__(self, length, add_dry=True, mask_ir=True, normalization_type=None, 
                 n_fft=2048, psd_prior="pink", psd_ema_alpha=0.99):
        super().__init__()
        self.length = length
        self.add_dry = add_dry
        self.mask_ir = mask_ir
        self.normalization_type = normalization_type
        self.n_fft = n_fft
        self.psd_ema_alpha = psd_ema_alpha
        self.ir = nn.Parameter(torch.randn(length) * 1e-6)
        
        if normalization_type == "psd_weighted_rms":
            self._initialize_spectrum_prior(psd_prior)

    def _initialize_spectrum_prior(self, psd_prior):
        """Initialize expected input power spectrum."""
        freqs = torch.fft.rfftfreq(self.n_fft)[1:]
        if psd_prior == "pink":
            S_x = 1.0 / (freqs + 1e-3)
        elif psd_prior == "white":
            S_x = torch.ones_like(freqs)
        else:
            S_x = 1.0 / (freqs + 1e-3)
        S_x = S_x / S_x.sum()
        self.register_buffer('S_x', S_x)

    def _update_spectrum_estimate(self, x):
        """Update running estimate of input PSD from batch."""
        if not self.training or self.normalization_type != "psd_weighted_rms":
            return
        X = torch.fft.rfft(x, n=self.n_fft, dim=-1)
        batch_psd = (X.abs() ** 2).mean(dim=0)[1:]
        batch_psd = batch_psd / (batch_psd.sum() + 1e-8)
        with torch.no_grad():
            self.S_x.data = self.psd_ema_alpha * self.S_x.data + (1 - self.psd_ema_alpha) * batch_psd

    def _normalize_rms(self, ir):
        """Apply simple RMS energy normalization."""
        energy = torch.sqrt((ir ** 2).sum(dim=-1, keepdim=True))
        return ir / (energy + 1e-8)

    def _normalize_psd_weighted(self, ir):
        """Apply PSD-weighted RMS normalization."""
        H = torch.fft.rfft(ir, n=self.n_fft)
        H_mag_sq = (H.abs() ** 2)[:, 1:]
        S_x = self.S_x.unsqueeze(0)
        numerator = S_x.sum(dim=-1, keepdim=True)
        denominator = (H_mag_sq * S_x).sum(dim=-1, keepdim=True)
        alpha = torch.sqrt(numerator / (denominator + 1e-8))
        return ir * alpha.unsqueeze(-1)

    def _apply_normalization(self, ir):
        """Apply selected normalization type."""
        if self.normalization_type == "rms":
            return self._normalize_rms(ir)
        elif self.normalization_type == "psd_weighted_rms":
            return self._normalize_psd_weighted(ir)
        return ir

    def mask_dry_ir(self, ir):
        """Forces the first tap (direct gain) of the IR to zero."""
        if len(ir.shape) == 1:
            ir = ir.unsqueeze(0)
        dry_mask = torch.zeros((ir.shape[0], 1), device=ir.device)
        return torch.cat([dry_mask, ir[:, 1:]], dim=1)

    def match_dimensions(self, audio, ir):
        """Ensures IR has batch dimension matching audio."""
        if len(ir.shape) == 1:
            ir = ir.unsqueeze(0)
        return ir.repeat(audio.shape[0], 1)

    def forward(self, x):
        original_shape = x.shape
        if len(x.shape) == 3:
            x = x.squeeze(-1)

        if self.normalization_type == "psd_weighted_rms":
            self._update_spectrum_estimate(x)

        ir = self.match_dimensions(x, self.ir)
        ir = self._apply_normalization(ir)
        
        if self.mask_ir:
            ir = self.mask_dry_ir(ir)

        wet_signal = fft_convolve(x, ir, delay_compensation=0)
        output = wet_signal + x if self.add_dry else wet_signal

        if len(original_shape) == 3:
            output = output.unsqueeze(-1)
        return output

    def build_impulse(self, apply_window=False):
        """Returns the learned impulse response, optionally windowed."""
        if apply_window:
            window = torch.hann_window(len(self.ir), device=self.ir.device)
            return self.ir * window
        return self.ir


class ARFilter(nn.Module):
    """Autoregressive filter using reflection coefficients for guaranteed stability."""
    
    def __init__(self, lpc_order=16, window_size=4096, add_dry=False, max_reflection=0.999, normalization_type=None):
        super().__init__()

        self.reflection_coeffs = nn.Parameter(torch.zeros(lpc_order))
        self.gain = nn.Parameter(torch.ones(1) * 0.1)
        self.max_reflection = max_reflection
        self.normalization_type = normalization_type

        window = torch.hann_window(window_size)
        self.register_buffer('_kernel', torch.diag(window).unsqueeze(1), persistent=False)

        self.add_dry = add_dry
        self.window_size = window_size
        self.hop_length = window_size // 4

    def reflection_to_direct(self, k):
        """Converts reflection coefficients to direct AR coefficients."""
        full_coeffs = reflection_to_ar(k)
        return full_coeffs[1:]

    def _apply_gain_normalization(self):
        """Normalize gain to achieve unit energy impulse response."""
        if self.normalization_type != "rms":
            return 1.0
        with torch.no_grad():
            k = self.max_reflection * torch.tanh(self.reflection_coeffs)
            ar_coeffs = self.reflection_to_direct(k)
            a_coeffs = torch.cat([torch.ones(1, device=ar_coeffs.device), ar_coeffs])
            b_coeffs = torch.zeros_like(a_coeffs)
            b_coeffs[0] = self.gain
            impulse = torch.zeros(self.window_size, device=self.reflection_coeffs.device)
            impulse[0] = 1.0
            impulse_response = lfilter(impulse, a_coeffs, b_coeffs)
            energy = torch.sqrt((impulse_response ** 2).sum())
            scale = 1.0 / (energy + 1e-8)
        return scale

    def forward(self, x):
        original_shape = x.shape
        if len(x.shape) == 3:
            x = x.squeeze(-1)

        batch_size, signal_length = x.shape

        k = self.max_reflection * torch.tanh(self.reflection_coeffs)
        ar_coeffs = self.reflection_to_direct(k)
        a_coeffs = torch.cat([torch.ones(1, device=ar_coeffs.device), ar_coeffs])

        gain_scale = self._apply_gain_normalization()
        b_coeffs = torch.zeros_like(a_coeffs)
        b_coeffs[0] = self.gain * gain_scale

        output = self._process_with_overlap_add(x, a_coeffs, b_coeffs, signal_length)

        if self.add_dry:
            output = output + x

        if len(original_shape) == 3:
            output = output.unsqueeze(-1)

        return output

    def _process_with_overlap_add(self, x, a_coeffs, b_coeffs, signal_length):
        """Process signal using overlap-add with IIR filtering."""
        padding = self.window_size // 2
        x_padded = F.pad(x, (padding, padding), 'constant', 0)
        unfolded = x_padded.unfold(-1, self.window_size, self.hop_length)

        batch, frames, window = unfolded.shape
        unfolded_flat = unfolded.reshape(-1, window)

        filtered = lfilter(unfolded_flat, a_coeffs.expand(batch*frames, -1),
                          b_coeffs.expand(batch*frames, -1))

        filtered = filtered.reshape(batch, frames, window).transpose(1, 2)

        ones = torch.ones(1, filtered.shape[1], filtered.shape[2], device=filtered.device)
        combined = torch.cat([filtered, ones], dim=0)
        result = F.conv_transpose1d(
            combined, self._kernel, stride=self.hop_length, padding=0
        ).squeeze(1)

        output, normalization = result[:-1], result[-1:]
        output = output / (normalization + 1e-8)

        if output.shape[1] > signal_length:
            output = output[:, :signal_length]
        elif output.shape[1] < signal_length:
            output = F.pad(output, (0, signal_length - output.shape[1]))

        return output

    def get_impulse_response(self, apply_window=False):
        """Computes the impulse response of the current AR filter."""
        k = self.max_reflection * torch.tanh(self.reflection_coeffs)
        ar_coeffs = self.reflection_to_direct(k)
        a_coeffs = torch.cat([torch.ones(1, device=ar_coeffs.device), ar_coeffs])
        
        b_coeffs = torch.zeros_like(a_coeffs)
        b_coeffs[0] = self.gain

        impulse = torch.zeros(self.window_size, device=self.reflection_coeffs.device)
        impulse[0] = 1.0

        impulse_response = lfilter(impulse, a_coeffs, b_coeffs)

        if apply_window:
            window = self._kernel.squeeze().diag()
            impulse_response = impulse_response * window[:impulse_response.shape[0]]

        return impulse_response

    def build_impulse(self, apply_window=False):
        """Interface method for consistency."""
        return self.get_impulse_response(apply_window=apply_window)


class ARMAFilter(nn.Module):
    """Autoregressive Moving Average filter using reflection coefficients for AR stability."""
    
    def __init__(self, ar_order: int = 16, ma_order: int = 16, window_size: int = 4096, add_dry: bool = False, max_reflection: float = 0.999, normalization_type=None):
        super().__init__()

        self.ar_order = ar_order
        self.ma_order = ma_order
        self.add_dry = add_dry
        self.max_reflection = max_reflection
        self.normalization_type = normalization_type

        self.ar_reflection_coeffs = nn.Parameter(torch.zeros(ar_order))
        self.ma_coeffs = nn.Parameter(torch.randn(ma_order) * 1e-4)
        self.gain = nn.Parameter(torch.ones(1) * 0.1)

        window = torch.hann_window(window_size)
        self.register_buffer('_kernel', torch.diag(window).unsqueeze(1), persistent=False)

        self.window_size = window_size
        self.hop_length = window_size // 4

    def reflection_to_ar(self, k):
        """Converts reflection coefficients to direct AR coefficients."""
        full_coeffs = reflection_to_ar(k)
        return full_coeffs[1:]

    def _apply_gain_normalization(self):
        """Normalize gain to achieve unit energy impulse response."""
        if self.normalization_type != "rms":
            return 1.0
        with torch.no_grad():
            k = self.max_reflection * torch.tanh(self.ar_reflection_coeffs)
            ar_coeffs = self.reflection_to_ar(k)
            a_coeffs = torch.cat([torch.ones(1, device=ar_coeffs.device), ar_coeffs])
            b_coeffs = torch.cat([self.gain, self.ma_coeffs])
            impulse = torch.zeros(self.window_size, device=ar_coeffs.device)
            impulse[0] = 1.0
            impulse_response = lfilter(impulse, a_coeffs, b_coeffs)
            energy = torch.sqrt((impulse_response ** 2).sum())
            scale = 1.0 / (energy + 1e-8)
        return scale

    def forward(self, x):
        original_shape = x.shape
        if len(x.shape) == 3:
            x = x.squeeze(-1)

        batch_size, signal_length = x.shape

        k = self.max_reflection * torch.tanh(self.ar_reflection_coeffs)
        ar_coeffs = self.reflection_to_ar(k)

        a_coeffs = torch.cat([torch.ones(1, device=ar_coeffs.device), ar_coeffs])
        
        gain_scale = self._apply_gain_normalization()
        b_coeffs = torch.cat([self.gain * gain_scale, self.ma_coeffs])

        output = self._process_with_overlap_add(x, a_coeffs, b_coeffs, signal_length)

        if self.add_dry:
            output = output + x

        if len(original_shape) == 3:
            output = output.unsqueeze(-1)

        return output

    def _process_with_overlap_add(self, x, a_coeffs, b_coeffs, signal_length):
        """Process signal using overlap-add with ARMA filtering."""
        padding = self.window_size // 2
        x_padded = F.pad(x, (padding, padding), 'constant', 0)
        unfolded = x_padded.unfold(-1, self.window_size, self.hop_length)

        batch, frames, window = unfolded.shape
        unfolded_flat = unfolded.reshape(-1, window)

        filtered = lfilter(unfolded_flat, a_coeffs.expand(batch*frames, -1),
                          b_coeffs.expand(batch*frames, -1))

        filtered = filtered.reshape(batch, frames, window).transpose(1, 2)

        ones = torch.ones(1, filtered.shape[1], filtered.shape[2], device=filtered.device)
        combined = torch.cat([filtered, ones], dim=0)
        result = F.conv_transpose1d(
            combined, self._kernel, stride=self.hop_length, padding=0
        ).squeeze(1)

        output, normalization = result[:-1], result[-1:]
        output = output / (normalization + 1e-8)

        if output.shape[1] > signal_length:
            output = output[:, :signal_length]
        elif output.shape[1] < signal_length:
            output = F.pad(output, (0, signal_length - output.shape[1]))

        return output

    def build_impulse(self, apply_window=False):
        """Computes the impulse response of the current ARMA filter."""
        k = self.max_reflection * torch.tanh(self.ar_reflection_coeffs)
        ar_coeffs = self.reflection_to_ar(k)
        a_coeffs = torch.cat([torch.ones(1, device=ar_coeffs.device), ar_coeffs])
        b_coeffs = torch.cat([self.gain, self.ma_coeffs])

        impulse = torch.zeros(self.window_size, device=ar_coeffs.device)
        impulse[0] = 1.0

        impulse_response = lfilter(impulse, a_coeffs, b_coeffs)

        if apply_window:
            window = self._kernel.squeeze().diag()
            impulse_response = impulse_response * window[:impulse_response.shape[0]]

        return impulse_response