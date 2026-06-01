import math

import torch
import torch.nn as nn

from reasoning_module import ReasoningModule
from utilities import trunc_normal_


class HierarchicalReasoningModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
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

        # ── f_I: Input network ───────────────────────────────────────────────
        # Embeds token indices into continuous vectors, then adds positional
        # information. Scaling by √D stabilises the embedding magnitudes.
        self.embed_scale = math.sqrt(hidden_size)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_pos = nn.Embedding(seq_len, hidden_size)  # learned positions

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

        self.zL_init = nn.Buffer(
            trunc_normal_(torch.empty(hidden_size, dtype=torch.bfloat16), std=1.0),
            persistent=True,
        )  # NOTE: hardcoded dtype.

    def compute_bp_steps(self, step: int, total_steps: int) -> int:
        warmup = total_steps * self.bp_warmup_ratio
        progress = min(1.0, step / warmup) if warmup > 0 else 1.0
        return self.bp_min_steps + int(
            progress * (self.bp_max_steps - self.bp_min_steps)
        )

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        z_H: torch.Tensor,  # (B, S, D)     — high-level carry from last call
        z_L: torch.Tensor,  # (B, S, D)     — low-level carry from last call
        x: torch.Tensor,  # (B, S)        — input token indices
        attn_mask: torch.Tensor | None = None,
        bp_steps: int = 5,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # Initialise recurrent states.
        z_H = x  # z_H₀ from token embeddings
        z_L = self.zL_init

        # Distribute bp_steps between H and L levels.
        # H is prioritised; L receives at least 1 step.
        H_bp = min(self.H_cycles, bp_steps - 1)
        L_bp = bp_steps - H_bp

        for i in range(self.H_cycles):
            # ── Fast L-level inner loop ────────────────────────────────────────
            for k in range(i * self.L_cycles, (i + 1) * self.L_cycles):
                # Enable gradients only for the last L_bp L-steps.
                # torch.is_grad_enabled() prevents re-enabling inside torch.no_grad().
                grad_on = torch.is_grad_enabled() and (
                    k >= self.H_cycles * self.L_cycles - L_bp
                )
                with torch.set_grad_enabled(grad_on):
                    z_L = self.L_net(z_L + z_H, attn_mask=attn_mask)

            # ── Slow H-level update ────────────────────────────────────────────
            # Enable gradients only for the last H_bp H-steps.
            grad_on = torch.is_grad_enabled() and (i >= self.H_cycles - H_bp)
            with torch.set_grad_enabled(grad_on):
                z_H = self.H_net(z_H + z_L, attn_mask=attn_mask)

        return z_H  # final high-level state → LM head

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
