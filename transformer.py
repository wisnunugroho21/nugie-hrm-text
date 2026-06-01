import torch
import torch.nn as nn

from attention import SigmoidGatedAttention
from rmsnorm import RMSNorm
from swiglu import SwiGLUFeedForward


class TransformerBlock(nn.Module):
    """
    A single Transformer block using Pre-Layer Normalization (PreNorm).

    PreNorm applies RMSNorm *before* each sublayer, then adds the sublayer's
    output back to the input via a residual connection:

        x ← x + Sublayer( RMSNorm(x) )

    This keeps the residual stream unnormalized, which creates a direct
    gradient highway from the output back to early layers — important for
    stable training when the block is used inside a recurrent loop.

    Each block contains two sublayers:
      1. Self-attention (SigmoidGatedAttention with RoPE and GQA)
      2. Feed-forward network (SwiGLU)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        norm_eps: float = 1e-6,
        expansion: float = 4 / 3,
    ):
        super().__init__()
        self.attn = SigmoidGatedAttention(
            hidden_size, num_heads, num_kv_heads, max_seq_len
        )
        self.mlp = SwiGLUFeedForward(hidden_size, expansion)

        # Separate norms for each sublayer (applied before the sublayer).
        self.attn_norm = RMSNorm(norm_eps)  # normalizes input before attention
        self.mlp_norm = RMSNorm(norm_eps)   # normalizes input before feed-forward

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Attention sublayer: normalize, attend, then add residual.
        x = x + self.attn(self.attn_norm(x), attn_mask=attn_mask)

        # Feed-forward sublayer: normalize, transform, then add residual.
        x = x + self.mlp(self.mlp_norm(x))

        return x
