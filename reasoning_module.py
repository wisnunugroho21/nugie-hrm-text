import torch
import torch.nn as nn

from rmsnorm import RMSNorm
from transformer import TransformerBlock


class ReasoningModule(nn.Module):
    """
    A recurrent reasoning module — either the High-level (H) or Low-level (L)
    module in the HRM hierarchy.

    Internally, it is a stack of TransformerBlocks (PreNorm) followed by a
    single boundary RMSNorm at the exit. This boundary norm is called
    "MagicNorm" in the paper because it simultaneously provides:

      - Forward stability: re-normalizes the hidden state after each call,
        bounding variance so it doesn't blow up over many recurrent steps.
      - Backward stability: with Truncated BPTT (small K), gradients only
        pass through this norm K times. Within the TBPTT window, gradients
        flow primarily through the PreNorm identity residual shortcuts,
        avoiding vanishing gradients.

    This module is called repeatedly in a loop by HierarchicalReasoningModel:
      - L_net is called L_cycles times per H cycle (fast, detail-oriented).
      - H_net is called once per H cycle (slow, high-level planning).
    """

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

        # Stack of PreNorm Transformer blocks — the core computation per step.
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

        # The "MagicNorm" boundary: applied once after all layers, at the
        # module boundary between recurrent steps.
        self.boundary_norm = RMSNorm(norm_eps)

    def forward(
        self,
        x: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pass the input through each transformer block sequentially.
        for block in self.layers:
            x = block(x, attn_mask=attn_mask)

        # Apply the boundary norm to stabilize the output before it is fed
        # back into the next recurrent step.
        return self.boundary_norm(x)
