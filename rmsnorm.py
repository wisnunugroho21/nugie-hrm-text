import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Unlike standard LayerNorm, RMSNorm normalizes by the root mean square of
    the activations rather than the mean and variance. It also omits the
    learnable scale (γ) and shift (β) parameters on purpose — keeping it a
    pure variance-control operation.

    Formula:  x̂ = x / sqrt( mean(x²) + ε )

    The small ε (default 1e-6) prevents division by zero when x is near zero.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps  # small constant for numerical stability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in float32 for numerical stability (important for bfloat16 training),
        # then cast back to the input dtype — matching the official HRM-Text implementation.
        original_dtype = x.dtype
        x = x.to(torch.float32)  # Convert to float32 for stable RMS calculation

        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # Normalize by RMS        
        return x.to(original_dtype)
