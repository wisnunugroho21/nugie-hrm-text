import torch
import torch.nn as nn

from attention import SigmoidGatedAttention
from rmsnorm import RMSNorm
from swiglu import SwiGLUFeedForward


class TransformerBlock(nn.Module):
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
        self.attn_norm = RMSNorm(norm_eps)  # applied *before* attention
        self.mlp_norm = RMSNorm(norm_eps)  # applied *before* feed-forward

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attn_mask=attn_mask)  # attention sublayer
        x = x + self.mlp(self.mlp_norm(x))  # feed-forward sublayer
        return x
