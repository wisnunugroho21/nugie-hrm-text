import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Instead of adding a fixed position vector to the token embedding, RoPE
    *rotates* pairs of head-dimension channels by a position-dependent angle.
    This means the dot product Q·Kᵀ naturally encodes the *relative* distance
    between two tokens — the model learns to be position-aware without needing
    a separate learned position table.

    How it works:
      - For each pair of channels (x_{2i}, x_{2i+1}), compute an angle
            θ_i = position / theta^(2i / head_dim)
        where theta=10_000 is the base frequency (like the Transformer paper).
      - Apply a 2D rotation by that angle to the pair.
      - This is equivalent to complex-number multiplication in the frequency
        domain, but done with real tensors via the rotate-half trick.

    Reference: RoFormer (Su et al., 2021).
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0):
        super().__init__()

        # Compute one inverse frequency per channel pair: shape [head_dim / 2].
        # Lower indices → higher frequencies (rotate faster with position).
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )

        # Outer product of positions [0..T-1] and frequencies → [T, head_dim/2].
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [T, head_dim / 2]

        # Duplicate along the last dim so it covers both halves: [T, head_dim].
        emb = torch.cat([freqs, freqs], dim=-1)

        # Register as non-persistent buffers: moved to the correct device with
        # the model, but not saved in state_dict (they are fully reproducible
        # from theta, so storing them would waste space).
        self.cos_table = nn.Buffer(emb.cos(), persistent=False)
        self.sin_table = nn.Buffer(emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to a query or key tensor.

        x: [B, T, n_heads, head_dim]
        Returns the same shape with rotated channel pairs.
        """
        B, T, n_heads, head_dim = x.shape

        # Slice to the actual sequence length and broadcast over B and heads.
        cos = self.cos_table[:T].unsqueeze(0).unsqueeze(2)  # → [1, T, 1, head_dim]
        sin = self.sin_table[:T].unsqueeze(0).unsqueeze(2)

        # Rotate-half trick: for each pair (x1, x2) apply [x1, x2] * cos + [-x2, x1] * sin.
        x1, x2 = x.chunk(2, dim=-1)
        return (x * cos + torch.cat([-x2, x1], dim=-1) * sin).to(x.dtype)
