import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities import trunc_normal_


class SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_size: int, expansion: float = 4 / 3):
        super().__init__()

        # Intermediate dimension chosen to match parameter budget of 4× FFN
        inter_size = int(round(expansion * hidden_size * 2 / 3))

        # Single matrix projects to 2*inter; first half = gate, second = value
        self.gate_up_proj = nn.Linear(hidden_size, inter_size * 2, bias=False)
        self.down_proj = nn.Linear(inter_size, hidden_size, bias=False)

        std_in = 1.0 / math.sqrt(hidden_size)
        std_out = 1.0 / math.sqrt(inter_size)
        trunc_normal_(self.gate_up_proj.weight, std=std_in)
        trunc_normal_(self.down_proj.weight, std=std_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)  # silu(gate) is the soft gate
