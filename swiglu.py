import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities import trunc_normal


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network (FFN).

    A drop-in replacement for the standard two-layer FFN that uses a gating
    mechanism inspired by GLU (Gated Linear Units):

        output = down_proj( SiLU(gate) × up )

    where `gate` and `up` are two separate projections of the input, obtained
    cheaply from a single fused matrix (gate_up_proj).

    Why SwiGLU?
      - SiLU(gate) acts as a smooth, data-dependent soft gate that selectively
        amplifies or suppresses each dimension of `up`.
      - This outperforms plain ReLU or GELU FFNs at the same parameter count.

    The intermediate dimension is scaled so the total parameter count matches a
    conventional 4× FFN: inter_size ≈ (2/3) × expansion × hidden_size.

    Reference: "GLU Variants Improve Transformer" (Shazeer, 2020).
    """

    def __init__(self, hidden_size: int, expansion: float = 4 / 3):
        super().__init__()

        # Compute the intermediate size to match a conventional 4× FFN budget.
        # The factor 2/3 compensates for the two-branch (gate + up) structure.
        inter_size = int(round(expansion * hidden_size * 2 / 3))

        # One fused projection outputs 2 × inter_size:
        # first half  → gate   (passed through SiLU)
        # second half → up     (passed through directly)
        self.gate_up_proj = nn.Linear(hidden_size, inter_size * 2, bias=False)

        # Projects the gated intermediate back down to hidden_size.
        self.down_proj = nn.Linear(inter_size, hidden_size, bias=False)

        # LeCun-style initialisation: scale by 1/√fan_in for each projection.
        std_in = 1.0 / math.sqrt(hidden_size)
        std_out = 1.0 / math.sqrt(inter_size)
        trunc_normal(self.gate_up_proj.weight, std=std_in)
        trunc_normal(self.down_proj.weight, std=std_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split the fused projection into the gate and value branches.
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)

        # SiLU(gate) is the "swish" soft gate; multiply element-wise with up.
        return self.down_proj(F.silu(gate) * up)
