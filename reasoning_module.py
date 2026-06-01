import torch
import torch.nn as nn

from rmsnorm import RMSNorm
from transformer import TransformerBlock


class ReasoningModule(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        norm_eps: float = 1e-6,
        expansion: float = 4 / 3,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_heads,
                    num_kv_heads,
                    max_seq_len,
                    norm_eps,
                    expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.boundary_norm = RMSNorm(norm_eps)  # this is the "MagicNorm" boundary

    def forward(
        self,
        x: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.layers:
            x = block(x, attn_mask=attn_mask)

        return self.boundary_norm(x)
