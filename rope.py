import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0):
        super().__init__()
        # One frequency per pair of channels: [head_dim / 2]
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [T, head_dim / 2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, head_dim]

        # Non-persistent: not saved in state_dict (recomputed from theta).
        self.register_buffer("cos_table", emb.cos(), persistent=False)
        self.register_buffer("sin_table", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, n_heads, head_dim = x.shape

        cos = self.cos_table[:T].unsqueeze(0).unsqueeze(2)  # → [1, T, 1, head_dim]
        sin = self.sin_table[:T].unsqueeze(0).unsqueeze(2)
        x1, x2 = x.chunk(2, dim=-1)

        return (x * cos + torch.cat([-x2, x1], dim=-1) * sin).to(x.dtype)
