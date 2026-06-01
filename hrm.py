import torch
import torch.nn as nn

from reasoning_module import ReasoningModule
from utilities import trunc_normal_


class HierarchicalReasoningModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        seq_len: int,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        H_layers: int = 2,
        L_layers: int = 2,
        H_cycles: int = 3,  # high-level cycles
        L_cycles: int = 3,  # low-level steps per cycle
        norm_eps: float = 1e-6,
        expansion: float = 4 / 3,
        bp_warmup_ratio: float = 0.2,  # Fraction of total steps for the warmup phase
        bp_min_steps: int = 2,  # TBPTT steps at the start of training  (K = 2)
        bp_max_steps: int = 5,  # TBPTT steps at the end of warmup      (K = 5)
    ):
        super().__init__()
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        self.bp_warmup_ratio = bp_warmup_ratio
        self.bp_min_steps = bp_min_steps
        self.bp_max_steps = bp_max_steps

        # ── f_L: Low-level recurrent module ─────────────────────────────────
        # Fast, detailed computation. Runs T times per high-level cycle.
        # Receives context = z_H + x̃  at every step.
        self.L_net = ReasoningModule(
            L_layers, hidden_size, num_heads, num_kv_heads, seq_len, norm_eps, expansion
        )

        # ── f_H: High-level recurrent module ────────────────────────────────
        # Slow, abstract planning. Runs once per T low-level steps.
        # Receives context = z_L  (L's converged output) once per cycle.
        self.H_net = ReasoningModule(
            H_layers, hidden_size, num_heads, num_kv_heads, seq_len, norm_eps, expansion
        )

        # Learned initial state for z_L: shape [D], broadcast to [B, T, D] in forward.
        # z_H is initialised from input embeddings so needs no separate init.
        zL_init = torch.empty(hidden_size)
        trunc_normal_(zL_init, std=1.0)
        self.zL_init = nn.Parameter(zL_init)

    def compute_bp_steps(self, step: int, total_steps: int) -> int:
        warmup = total_steps * self.bp_warmup_ratio
        progress = min(1.0, step / warmup) if warmup > 0 else 1.0
        return self.bp_min_steps + int(
            progress * (self.bp_max_steps - self.bp_min_steps)
        )

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,  # (B, T, D) — input embeddings
        attn_mask: torch.Tensor | None = None,
        bp_steps: int = 5,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # Initialise recurrent states.
        z_H = x  # z_H₀ from token embeddings
        z_L = self.zL_init[None, None, :].expand(B, T, -1)  # broadcast [D] → [B, T, D]

        total_L = self.H_cycles * self.L_cycles  # total L-module calls

        # Distribute bp_steps between H and L levels.
        # H is prioritised; L receives at least 1 step.
        H_bp = min(self.H_cycles, bp_steps - 1)
        L_bp = bp_steps - H_bp

        l_step = 0  # running index of L calls (used for TBPTT cutoff)

        for i in range(self.H_cycles):
            # ── Fast L-level inner loop ────────────────────────────────────────
            for _j in range(self.L_cycles):
                # Enable gradients only for the last L_bp L-steps.
                # torch.is_grad_enabled() prevents re-enabling inside torch.no_grad().
                grad_on = torch.is_grad_enabled() and (l_step >= total_L - L_bp)
                with torch.set_grad_enabled(grad_on):
                    z_L = self.L_net(z_L + z_H, attn_mask=attn_mask)
                l_step += 1

            # ── Slow H-level update ────────────────────────────────────────────
            # Enable gradients only for the last H_bp H-steps.
            grad_on = torch.is_grad_enabled() and (i >= self.H_cycles - H_bp)
            with torch.set_grad_enabled(grad_on):
                z_H = self.H_net(z_H + z_L, attn_mask=attn_mask)

        return z_H  # final high-level state → LM head

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
