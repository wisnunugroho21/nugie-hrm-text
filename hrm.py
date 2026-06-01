import torch
import torch.nn as nn

from reasoning_module import ReasoningModule
from utilities import trunc_normal_


class HierarchicalReasoningModel(nn.Module):
    """
    Hierarchical Reasoning Model (HRM) core — the dual-timescale recurrent
    backbone described in the HRM-Text paper.

    Architecture overview
    ─────────────────────
    HRM replaces the single forward pass of a standard Transformer with a
    nested recurrent loop of two modules:

      • H (High-level, "slow"):  Maintains broad semantic / planning context.
                                 Updated once per outer cycle.
      • L (Low-level, "fast"):   Performs fine-grained local refinement.
                                 Updated L_cycles times per outer cycle.

    Recurrent loop (pseudocode):
        z_H = x   (initialized from input embeddings)
        z_L = learned_init  (trainable parameter, broadcast to [B, T, D])

        for i in range(H_cycles):
            for j in range(L_cycles):
                z_L = L_net(z_L + z_H)   # L receives z_H as high-level context
            z_H = H_net(z_H + z_L)       # H receives z_L as low-level summary

    After all cycles, z_H is the final hidden state passed to the LM head.

    Parameter sharing:
        The same L_net weights are reused at every L step, and the same
        H_net weights at every H step — similar to a weight-tied RNN.
        Total parameter count matches a single-pass Transformer of the same
        layer budget (each module gets num_layers // 2 layers).

    Truncated BPTT (TBPTT):
        Training through the full recurrent loop is expensive and can lead to
        vanishing gradients. Instead, only the *last bp_steps recurrent calls*
        are backpropagated through. Earlier steps are run under torch.no_grad()
        to detach their gradients. bp_steps is linearly warmed up from
        bp_min_steps to bp_max_steps over the first bp_warmup_ratio of training.
    """

    def __init__(
        self,
        hidden_size: int,
        seq_len: int,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        H_layers: int = 2,
        L_layers: int = 2,
        H_cycles: int = 3,  # number of outer (high-level) cycles
        L_cycles: int = 3,  # number of inner (low-level) steps per cycle
        norm_eps: float = 1e-6,
        expansion: float = 4 / 3,
        bp_warmup_ratio: float = 0.2,  # fraction of total training steps for TBPTT warmup
        bp_min_steps: int = 2,         # TBPTT window at the start of training
        bp_max_steps: int = 5,         # TBPTT window at the end of warmup
    ):
        super().__init__()
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        # TBPTT schedule parameters — stored for compute_bp_steps().
        self.bp_warmup_ratio = bp_warmup_ratio
        self.bp_min_steps = bp_min_steps
        self.bp_max_steps = bp_max_steps

        # f_L: Low-level recurrent module.
        # Fast, fine-grained computation. Runs L_cycles times per H cycle.
        # At each step it receives z_L + z_H as input, so H provides context.
        self.L_net = ReasoningModule(
            L_layers, hidden_size, num_heads, num_kv_heads, seq_len, norm_eps, expansion
        )

        # f_H: High-level recurrent module.
        # Slow, abstract planning. Runs once per set of L_cycles steps.
        # At each step it receives z_H + z_L as input, so L provides a summary.
        self.H_net = ReasoningModule(
            H_layers, hidden_size, num_heads, num_kv_heads, seq_len, norm_eps, expansion
        )

        # Learned initial state for z_L (shape [D], broadcast to [B, T, D]).
        # z_H starts from the input embeddings so it doesn't need a separate init.
        zL_init = torch.empty(hidden_size)
        trunc_normal_(zL_init, std=1.0)
        self.zL_init = nn.Parameter(zL_init)

    def compute_bp_steps(self, step: int, total_steps: int) -> int:
        """
        Compute the TBPTT window size (K) for the current training step.

        K grows linearly from bp_min_steps to bp_max_steps over the first
        bp_warmup_ratio of training. After the warmup period, K stays at
        bp_max_steps for the rest of training.

        Starting with a small K is beneficial early in training when the
        recurrent states are noisy — shorter gradients are more stable and
        act as a form of curriculum.
        """
        warmup = total_steps * self.bp_warmup_ratio
        progress = min(1.0, step / warmup) if warmup > 0 else 1.0
        return self.bp_min_steps + int(
            progress * (self.bp_max_steps - self.bp_min_steps)
        )

    def forward(
        self,
        x: torch.Tensor,          # (B, T, D) — input embeddings
        attn_mask: torch.Tensor | None = None,
        bp_steps: int = 5,        # TBPTT window; use compute_bp_steps() during training
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # Initialize recurrent states.
        z_H = x  # H starts from the token embeddings (encodes input structure)
        z_L = self.zL_init[None, None, :].expand(B, T, -1)  # broadcast [D] → [B, T, D]

        total_L = self.H_cycles * self.L_cycles  # total number of L-module calls

        # Distribute bp_steps across the two levels:
        # H is prioritized; L gets at least 1 gradient step.
        H_bp = min(self.H_cycles, bp_steps - 1)
        L_bp = bp_steps - H_bp

        l_step = 0  # running count of L-module calls (used for TBPTT cutoff)

        for i in range(self.H_cycles):
            # --- Inner L-level loop -------------------------------------------
            for _j in range(self.L_cycles):
                # Only enable gradients for the last L_bp calls to L_net.
                # torch.is_grad_enabled() check avoids re-enabling grad inside
                # an outer torch.no_grad() context (e.g., during validation).
                grad_on = torch.is_grad_enabled() and (l_step >= total_L - L_bp)
                with torch.set_grad_enabled(grad_on):
                    # L receives the sum of its own state and the H state as context.
                    z_L = self.L_net(z_L + z_H, attn_mask=attn_mask)
                l_step += 1

            # --- Outer H-level update -----------------------------------------
            # Only enable gradients for the last H_bp calls to H_net.
            grad_on = torch.is_grad_enabled() and (i >= self.H_cycles - H_bp)
            with torch.set_grad_enabled(grad_on):
                # H receives the sum of its own state and the final z_L as a summary.
                z_H = self.H_net(z_H + z_L, attn_mask=attn_mask)

        # Return the final high-level state — fed to the LM head for next-token prediction.
        return z_H

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
