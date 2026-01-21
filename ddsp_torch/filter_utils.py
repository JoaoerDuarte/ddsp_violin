import torch
import numpy as np


def reflection_to_ar(reflection_coefficients):
    """Converts reflection coefficients (k) to direct-form AR coefficients (a)."""
    is_tensor = isinstance(reflection_coefficients, torch.Tensor)

    if is_tensor:
        k = reflection_coefficients.clone()
        device = k.device
        dtype = k.dtype
    else:
        k_array = np.asarray(reflection_coefficients)
        k = torch.tensor(k_array)
        device = k.device
        dtype = k.dtype

    # Ensure batch dimension for consistent processing
    batched = k.dim() > 1
    if not batched:
        k = k.unsqueeze(0)

    batch_size, Q = k.shape
    a = torch.zeros(batch_size, Q + 1, device=device, dtype=dtype)
    a[:, 0] = 1.0  # a_0 is always 1

    # Iteratively compute AR coefficients using Levinson-Durbin recursion
    for i in range(Q):
        a_previous = a[:, :i+1].clone()  # Coefficients from previous step
        a[:, i+1] = k[:, i] * a_previous[:, 0]

        if i > 0:
            # Update intermediate coefficients using recursion
            indices = torch.arange(1, i + 1, device=device)
            reversed_indices = i - indices + 1
            a[:, indices] = a_previous[:, indices] + k[:, i:i+1] * a_previous[:, reversed_indices]

    # Remove batch dimension if it was added
    if not batched:
        a = a.squeeze(0)
        
    return a