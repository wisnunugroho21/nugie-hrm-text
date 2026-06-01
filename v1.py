"""
hrm_text.py

Minimal PyTorch implementation of HRM-Text.

Paper : "HRM-Text: Efficient Pretraining Beyond Scaling"
        https://arxiv.org/abs/2605.20613
Code  : https://github.com/sapientinc/HRM-Text

────────────────────────────────────────────────────────────────────────────────
Overview
────────────────────────────────────────────────────────────────────────────────

HRM-Text replaces the standard Transformer with a Hierarchical Recurrent
Model (HRM), inspired by the dual-timescale processing found in the brain's
frontoparietal loop.

Key ideas:

  1. Dual-timescale recurrence (Section 2)
       Two modules run in a nested loop every forward pass:
         • H (high-level, "slow"): maintains strategic / semantic context.
         • L (low-level, "fast"): performs local iterative refinement.
       Default: 2 outer H cycles × 3 inner L cycles = H2L3 notation.
       Each module uses *half* of the total layer budget (half_layers=True),
       so total parameter count matches a single-pass Transformer.

  2. MagicNorm (Section 2.1.1)
       Each recurrent module is a stack of PreNorm Transformer blocks
       *topped by a final RMSNorm* at the module boundary.
         • Forward : the boundary norm bounds hidden-state variance each
                     recurrent step → PostNorm-like numerical stability.
         • Backward: with truncated BPTT (small K), gradients flow mainly
                     through the PreNorm residual shortcuts → PreNorm-like
                     gradient flow, avoiding vanishing gradients.

  3. Warmup deep credit assignment (Section 2.1.2)
       Backpropagation is truncated to the last K recurrent steps (TBPTT).
       K starts at bp_min_steps=2 and linearly warms up to bp_max_steps=5.
       Steps outside the TBPTT window run under torch.no_grad() so their
       tensors are detached, cutting off gradient flow.

  4. Task-completion objective (Section 2.2)
       Train on instruction–response pairs from scratch.
       Loss is computed only over *response* tokens.
       Instruction tokens receive label = -100 (PyTorch's ignore index).

  5. PrefixLM attention mask (Section 2.2)
       Instruction tokens attend to each other bidirectionally.
       Response tokens use standard causal masking.
       This gives the model encoder-like context over instructions while
       keeping autoregressive generation for responses.

Additional architecture details (Figure 2 in the paper):
  • Sigmoid-gated attention:  out = sigmoid(gate) × attn(Q, K, V)
  • SwiGLU feed-forward network
  • Rotary Position Embeddings (RoPE)
  • Parameterless RMSNorm (no learnable γ / β)
  • Scaled embeddings: embed_out = (1 / init_std) × lookup(ids)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class HRMConfig:
    """All hyper-parameters for an HRM-Text model."""

    vocab_size: int = 32_000

    # Transformer dimensions
    hidden_size: int = 512  # Token embedding / hidden-state width D
    num_heads: int = 8  # Attention heads  (head_dim = D / num_heads)
    num_layers: int = 8  # Transformer layers *before* halving
    ffn_hidden_size: int = 1024  # SwiGLU inner width (before gate split)

    # Sequence
    max_seq_len: int = 2048
    rope_theta: float = 10_000.0  # RoPE base frequency θ

    # Normalization
    norm_eps: float = 1e-6  # ε for RMSNorm

    # ── HRM recurrence ────────────────────────────────────────────────────────
    H_cycles: int = 2  # Outer slow-H cycles  (paper default: 2)
    L_cycles: int = 3  # Inner fast-L cycles per H cycle  (paper default: 3)
    half_layers: bool = True  # If True, each H/L module gets num_layers // 2 layers

    # ── Warmup deep credit assignment (truncated BPTT) ────────────────────────
    bp_warmup_ratio: float = 0.2  # Fraction of total steps for the warmup phase
    bp_min_steps: int = 2  # TBPTT steps at the start of training  (K = 2)
    bp_max_steps: int = 5  # TBPTT steps at the end of warmup      (K = 5)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Utilities
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    """
    Fast approximate truncated-normal initialisation (matches the official code).
    Draws from N(0,1), clamps to ±3, then rescales to the desired std.

    @torch.no_grad() allows safe in-place use on nn.Parameters (which have
    requires_grad=True and would otherwise raise on in-place ops).
    """
    return tensor.normal_().fmod_(3.0).mul_(1.014_762_601_732_121 * std)


class RMSNorm(nn.Module):
    """
    Parameterless RMSNorm — deliberately no learnable scale γ or shift β.

    The paper explicitly omits γ to keep the norm a pure variance-control
    operation, avoiding the extra interaction between γ and the residual stream.

    Formula:  x̂ = x / sqrt( mean(x²) + ε )
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Rotary Position Embedding (RoPE)
# ──────────────────────────────────────────────────────────────────────────────


class RotaryEmbedding(nn.Module):
    """
    Precomputes cos/sin tables for Rotary Position Embeddings.

    RoPE rotates pairs of head-dimension channels by position-dependent angles,
    so the Q·Kᵀ dot-product encodes the *relative* distance between tokens
    without requiring explicit position tokens or learned position matrices.

    Reference: RoFormer (Su et al., 2021).
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float):
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

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (cos, sin) each of shape [seq_len, head_dim]."""
        return self.cos_table[:seq_len], self.sin_table[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Splits x in half along the last dim and returns [-x₂, x₁]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to a query or key tensor.

    x:   [B, T, n_heads, head_dim]
    cos / sin: [T, head_dim]  — broadcast over B and n_heads dimensions.
    """
    cos = cos.unsqueeze(0).unsqueeze(2)  # → [1, T, 1, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(2)
    return (x * cos + _rotate_half(x) * sin).to(x.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Sigmoid-Gated Multi-Head Self-Attention
# ──────────────────────────────────────────────────────────────────────────────


class SigmoidGatedAttention(nn.Module):
    """
    Multi-head self-attention with an additional sigmoid gate (Figure 2c).

    Standard attention:
        out = softmax( Q·Kᵀ / √d ) @ V

    Sigmoid-gated attention adds a data-dependent output gate:
        out = sigmoid(gate) × attn(Q, K, V)

    gate, Q, K, V are all linearly projected from the same input in a single
    fused operation.  The gate lets the model suppress irrelevant information,
    similar to LSTM / GRU gating — but computed in one step.

    Reference: "Gated Attention for Large Language Models" (Zhu et al., 2024).
    """

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        D = config.hidden_size

        # One fused linear: [D] → [gate(D) | Q(D) | K(D) | V(D)] = 4D
        self.qkv_gate_proj = nn.Linear(D, 4 * D, bias=False)
        self.out_proj = nn.Linear(D, D, bias=False)

        # LeCun-normal initialisation (1 / √D), matching the official code.
        std = 1.0 / math.sqrt(D)
        trunc_normal_(self.qkv_gate_proj.weight, std=std)
        trunc_normal_(self.out_proj.weight, std=std)

        self.rope = RotaryEmbedding(
            self.head_dim, config.max_seq_len, config.rope_theta
        )
        self.scale = self.head_dim**-0.5  # 1/√head_dim scaling for attention

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x:         [B, T, D]
        attn_mask: [B, 1, T, T] additive float mask  (0.0 = attend, -inf = block)
                   or None → causal masking via is_causal=True.
        """
        B, T, D = x.shape

        # ── Project and split into gate / Q / K / V ───────────────────────────
        gate, q, k, v = self.qkv_gate_proj(x).chunk(4, dim=-1)  # each [B, T, D]

        # ── Reshape to [B, T, n_heads, head_dim] ─────────────────────────────
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
        gate = gate.view(B, T, self.num_heads, self.head_dim)

        # ── Apply RoPE to Q and K ─────────────────────────────────────────────
        cos, sin = self.rope(T)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # ── Scaled dot-product attention ──────────────────────────────────────
        # Transpose to [B, n_heads, T, head_dim] for PyTorch's SDPA.
        # SDPA dispatches to Flash Attention when available.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # is_causal=True only when there is no custom mask (mutually exclusive).
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=(attn_mask is None),
            scale=self.scale,
        )  # → [B, n_heads, T, head_dim]

        # ── Sigmoid gate ──────────────────────────────────────────────────────
        attn_out = attn_out.transpose(1, 2)  # → [B, T, n_heads, head_dim]
        gated = torch.sigmoid(gate) * attn_out  # element-wise gate
        gated = gated.reshape(B, T, D)  # flatten heads → [B, T, D]

        return self.out_proj(gated)


# ──────────────────────────────────────────────────────────────────────────────
# 5. SwiGLU Feed-Forward Network
# ──────────────────────────────────────────────────────────────────────────────


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward block.

    Projects to 2 × ffn_hidden_size, splits into (gate, up), then:
        output = down_proj( SiLU(gate) × up )

    The gating makes each activation dimension selectivable, while SiLU
    provides a smooth, non-zero gradient almost everywhere.

    Reference: "GLU Variants Improve Transformer" (Shazeer, 2020).
    """

    def __init__(self, config: HRMConfig):
        super().__init__()
        # Fused up + gate projection; will be split into two equal halves.
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * config.ffn_hidden_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.ffn_hidden_size, config.hidden_size, bias=False
        )

        std_in = 1.0 / math.sqrt(config.hidden_size)
        std_out = 1.0 / math.sqrt(config.ffn_hidden_size)
        trunc_normal_(self.gate_up_proj.weight, std=std_in)
        trunc_normal_(self.down_proj.weight, std=std_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Transformer Block (Pre-Layer Norm)
# ──────────────────────────────────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    """
    A single Transformer block using Pre-Layer Normalization (PreNorm).

    PreNorm formula:
        x ← x + Sublayer( RMSNorm(x) )

    By normalising *before* each sublayer, the residual stream remains
    unnormalised, creating a clean identity shortcut:
        x_L = x_0 + Σ_l Sublayer_l(...)

    This direct gradient path is important for MagicNorm's backward
    stability: gradients reach early steps through the identity connections
    without passing through boundary norms (which would shrink them).
    """

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.attn = SigmoidGatedAttention(config)
        self.mlp = SwiGLU(config)
        self.attn_norm = RMSNorm(config.norm_eps)  # applied *before* attention
        self.mlp_norm = RMSNorm(config.norm_eps)  # applied *before* feed-forward

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attn_mask=attn_mask)  # attention sublayer
        x = x + self.mlp(self.mlp_norm(x))  # feed-forward sublayer
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 7. Recurrent Module with MagicNorm
# ──────────────────────────────────────────────────────────────────────────────


class RecurrentModule(nn.Module):
    """
    A single recurrent module (H or L level) implementing MagicNorm (Figure 2b).

    MagicNorm = PreNorm Transformer stack + boundary RMSNorm at the exit.

        z_out = FinalNorm( forward_through_prenorm_stack( z_in + injection ) )

    Why the boundary norm matters:
      • Forward stability — at every recurrent step the state is re-normalised,
        bounding variance the way PostNorm would.  Without it, pure PreNorm
        stacks accumulate unbounded variance over many recurrent steps.
      • Backward stability — with truncated BPTT (small K), gradients only pass
        through this boundary norm K times.  Inside that window they flow mainly
        through the L PreNorm identity shortcuts.  This gives PreNorm-like
        gradient flow, avoiding vanishing gradients.

    Input injection:
        The other level's hidden state is *added* to this module's state
        before processing, coupling the two levels.  Simple addition is used
        here (the official code notes GRU-style gating as a future direction).
    """

    def __init__(self, config: HRMConfig, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(n_layers)])
        self.final_norm = RMSNorm(config.norm_eps)  # the "magic" boundary norm

    def forward(
        self,
        hidden: torch.Tensor,
        inject: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        hidden:    [B, T, D]  — this module's current recurrent state.
        inject:    [B, T, D]  — the other level's state (cross-level signal).
        attn_mask: float mask [B, 1, T, T] or None.
        """
        # Additive injection: merge the partner level's context before processing.
        hidden = hidden + inject

        # PreNorm Transformer stack (each block: x ← x + Sublayer(Norm(x))).
        for layer in self.layers:
            hidden = layer(hidden, attn_mask=attn_mask)

        # Boundary norm: bounds variance, provides MagicNorm's forward stability.
        return self.final_norm(hidden)


# ──────────────────────────────────────────────────────────────────────────────
# 8. HRM — Hierarchical Recurrent Model
# ──────────────────────────────────────────────────────────────────────────────


class HRM(nn.Module):
    """
    HRM core: orchestrates the nested H / L recurrence.

    Forward pass (H2L3 example, i.e. H_cycles=2, L_cycles=3):

        z_H ← token embeddings          (high-level state, initialised from input)
        z_L ← learned zL_init           (low-level state, fixed learned init)

        for i in 0..1:                   ← outer slow H cycle
            for j in 0..2:               ← inner fast L cycle
                z_L = L(z_L, inject=z_H)   L updated with H guidance
            z_H = H(z_H, inject=z_L)       H updated with L feedback

        return z_H  →  LM head

    Total module calls: H_cycles × (L_cycles + 1) = 2 × 4 = 8.
    Since H and L each use half the layer budget, effective compute per
    forward pass = 4 × a full-parameter single-pass Transformer.

    Warmup deep credit assignment (truncated BPTT):
        During each forward pass a bp_steps counter determines how many
        recurrent steps at the *end* of the loop receive gradients.
        Earlier steps run under torch.no_grad() — their outputs are
        detached, cutting the gradient path through those steps.

        bp_steps is scheduled to grow linearly from bp_min_steps=2 to
        bp_max_steps=5 over the first (bp_warmup_ratio × total_steps) steps.

        H steps are prioritised: H gets min(H_cycles, bp_steps-1) slots,
        L gets the remaining bp_steps − H_slots slots.
    """

    def __init__(self, config: HRMConfig):
        super().__init__()

        # Each module uses half the total layer budget when half_layers=True.
        n_per_module = (
            config.num_layers // 2 if config.half_layers else config.num_layers
        )

        # H and L modules: separate weights (not shared / not tied).
        self.H = RecurrentModule(config, n_per_module)
        self.L = RecurrentModule(config, n_per_module)

        self.H_cycles = config.H_cycles
        self.L_cycles = config.L_cycles
        self.bp_warmup_ratio = config.bp_warmup_ratio
        self.bp_min_steps = config.bp_min_steps
        self.bp_max_steps = config.bp_max_steps

        # Learned initial state for z_L.
        # z_H is initialised from input embeddings so it needs no separate init.
        # Shape: [D] — will be broadcast to [B, T, D] in forward().
        zL_init = torch.empty(config.hidden_size)
        trunc_normal_(zL_init, std=1.0)
        self.zL_init = nn.Parameter(zL_init)

    def compute_bp_steps(self, step: int, total_steps: int) -> int:
        """
        Returns the TBPTT window K for the current training step.

        Linearly interpolates from bp_min_steps to bp_max_steps over
        the first (bp_warmup_ratio × total_steps) gradient steps.

        Example schedule with bp_warmup_ratio=0.2, total_steps=1000:
            step   0:  K = 2
            step 100:  K = 3   (50% through warmup)
            step 200:  K = 5   (100% through warmup)
            step 500:  K = 5   (capped)
        """
        warmup = total_steps * self.bp_warmup_ratio
        progress = min(1.0, step / warmup) if warmup > 0 else 1.0
        return self.bp_min_steps + int(
            progress * (self.bp_max_steps - self.bp_min_steps)
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        bp_steps: int = 5,
    ) -> torch.Tensor:
        """
        x:         [B, T, D]        — token embeddings (initialises z_H).
        attn_mask: [B, 1, T, T]     — float additive mask, or None (→ causal).
        bp_steps:  int              — TBPTT window; call compute_bp_steps() during training.

        Returns z_H: [B, T, D]  — final high-level recurrent state.
        """
        B, T, _ = x.shape

        # Initialise recurrent states.
        z_H = x  # z_H₀ from token embeddings
        z_L = self.zL_init[None, None, :].expand(
            B, T, -1
        )  # z_L₀ broadcast to [B, T, D]

        total_L = self.H_cycles * self.L_cycles  # total L-module calls in one forward
        total_H = self.H_cycles  # total H-module calls in one forward

        # Distribute bp_steps between H and L levels.
        # H is prioritised; L receives at least 1 step.
        H_bp = min(total_H, bp_steps - 1)
        L_bp = bp_steps - H_bp

        l_step = 0  # running index of L calls (used for TBPTT cutoff)

        for i in range(self.H_cycles):
            # ── Fast L-level inner loop ────────────────────────────────────────
            for _j in range(self.L_cycles):
                # Enable gradients only for the last L_bp L-steps.
                # torch.is_grad_enabled() prevents re-enabling inside torch.no_grad().
                grad_on = torch.is_grad_enabled() and (l_step >= total_L - L_bp)
                with torch.set_grad_enabled(grad_on):
                    z_L = self.L(z_L, inject=z_H, attn_mask=attn_mask)
                l_step += 1

            # ── Slow H-level update ────────────────────────────────────────────
            # Enable gradients only for the last H_bp H-steps.
            grad_on = torch.is_grad_enabled() and (i >= total_H - H_bp)
            with torch.set_grad_enabled(grad_on):
                z_H = self.H(z_H, inject=z_L, attn_mask=attn_mask)

        return z_H  # final high-level state → LM head


# ──────────────────────────────────────────────────────────────────────────────
# 9. PrefixLM Attention Mask
# ──────────────────────────────────────────────────────────────────────────────


def make_prefixlm_mask(
    prefix_lens: torch.Tensor, total_len: int, device: torch.device
) -> torch.Tensor:
    """
    Build a PrefixLM attention mask for a batch of instruction–response sequences.

    Attention rules (Section 2.2, Figure 2d):
      • Instruction tokens (positions 0 … prefix_len-1):
            Bidirectional — each token can see all other instruction tokens.
      • Response tokens (positions prefix_len … T-1):
            Causal — each token attends only to itself and earlier tokens.

    This gives HRM-Text an encoder–decoder-like split inside a single
    decoder-style model:
      instruction = encoder-like (fully visible bidirectional context)
      response    = decoder-like (autoregressive generation)

    Args:
        prefix_lens: [B]  — instruction length per example.
        total_len:   int  — full sequence length (instruction + response).
        device:      where to create the mask tensor.

    Returns:
        [B, 1, T, T] additive float mask
            0.0   where attention is allowed
            -inf  where attention is blocked
        The head dimension of 1 broadcasts across all attention heads.
    """
    B, T = prefix_lens.size(0), total_len

    # Start with a lower-triangular causal mask (True = allowed to attend).
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    mask = causal.unsqueeze(0).expand(B, T, T).clone()  # [B, T, T]

    # Override: instruction tokens attend to all other instruction tokens.
    for b in range(B):
        plen = int(prefix_lens[b].item())
        mask[b, :plen, :plen] = True  # full bidirectional within the prefix

    # Convert boolean → additive float mask:
    #   True  (allowed) → 0.0
    #   False (blocked) → -inf
    float_mask = torch.zeros(B, 1, T, T, dtype=torch.float32, device=device)
    float_mask.masked_fill_(~mask.unsqueeze(1), float("-inf"))
    return float_mask


# ──────────────────────────────────────────────────────────────────────────────
# 10. HRMText — Full Language Model
# ──────────────────────────────────────────────────────────────────────────────


class HRMText(nn.Module):
    """
    HRM-Text: complete language model wrapping the HRM core.

    Pipeline:
        1. Token embedding  — look up embeddings and apply a scale factor.
        2. HRM recurrence   — nested H / L recurrent processing with MagicNorm.
        3. LM head          — linear projection to vocabulary logits.

    Scaled embeddings:
        The weight matrix is initialised with std = 1/√D.
        At runtime the output is multiplied by (1 / std) = √D.
        This keeps embedding norms roughly proportional to the hidden states
        produced by the model, improving training stability.

    Task-completion objective:
        Loss is computed only over *response* tokens.  Instruction tokens are
        excluded by setting their labels to IGNORE_LABEL = -100, which is
        the standard PyTorch ignore index for cross_entropy.

    ── Usage (training step) ──────────────────────────────────────────────────
        model = HRMText(HRMConfig(...))
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4)

        bp_steps = model.hrm.compute_bp_steps(step, total_steps)
        loss, logits = model(input_ids, labels=labels,
                             prefix_lens=prefix_lens, bp_steps=bp_steps)
        loss.backward()
        optimizer.step()

    ── Usage (inference) ──────────────────────────────────────────────────────
        with torch.no_grad():
            _, logits = model(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1)
    """

    IGNORE_LABEL: int = -100  # tokens with this label are excluded from the loss

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config

        # ── Token embedding (scaled) ──────────────────────────────────────────
        init_std = 1.0 / math.sqrt(config.hidden_size)  # LeCun std = 1/√D
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        with torch.no_grad():
            trunc_normal_(self.embed_tokens.weight, std=init_std)
        self.embed_scale = 1.0 / init_std  # runtime multiplier (= √D)

        # ── HRM core ──────────────────────────────────────────────────────────
        self.hrm = HRM(config)

        # ── Output projection (no bias) ────────────────────────────────────────
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        with torch.no_grad():
            trunc_normal_(self.lm_head.weight, std=init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        prefix_lens: Optional[torch.Tensor] = None,
        bp_steps: int = 5,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            input_ids:   [B, T]  — token IDs.
            labels:      [B, T]  — target token IDs.
                                   Set instruction positions to -100 for the
                                   task-completion objective (response-only loss).
            prefix_lens: [B]     — instruction token count per example.
                                   If provided, PrefixLM masking is applied;
                                   otherwise, standard causal masking is used.
            bp_steps:    int     — TBPTT window; use compute_bp_steps() during training.

        Returns:
            loss:   scalar cross-entropy loss, or None if labels is not provided.
            logits: [B, T, vocab_size].
        """
        B, T = input_ids.shape

        # 1. Token embedding with scaling: keeps embedding norms stable.
        x = self.embed_scale * self.embed_tokens(input_ids)  # [B, T, D]

        # 2. Build attention mask.
        if prefix_lens is not None:
            # PrefixLM: bidirectional attention over instruction tokens,
            #           causal masking over response tokens.
            attn_mask = make_prefixlm_mask(prefix_lens, T, input_ids.device)
        else:
            # No custom mask → SigmoidGatedAttention will use is_causal=True.
            attn_mask = None

        # 3. HRM recurrent forward pass.
        z_H = self.hrm(x, attn_mask=attn_mask, bp_steps=bp_steps)  # [B, T, D]

        # 4. Project to vocabulary logits.
        logits = self.lm_head(z_H)  # [B, T, vocab_size]

        # 5. Task-completion loss (response tokens only).
        loss = None
        if labels is not None:
            # Cast logits to float32 for stable loss computation (common practice).
            # Tokens with label == -100 are automatically skipped by cross_entropy.
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size).float(),
                labels.view(-1).long(),
                ignore_index=self.IGNORE_LABEL,
            )

        return loss, logits


# ──────────────────────────────────────────────────────────────────────────────
# Quick-start smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    # Tiny config for a quick forward/backward smoke test.
    cfg = HRMConfig(
        vocab_size=256,
        hidden_size=64,
        num_heads=4,
        num_layers=4,  # → 2 layers per H/L module (half_layers=True)
        ffn_hidden_size=128,
        max_seq_len=32,
        H_cycles=2,
        L_cycles=3,
    )

    model = HRMText(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Dummy instruction–response batch ──────────────────────────────────────
    B, T = 2, 16
    prefix_len = 6  # first 6 tokens = "instruction"; rest = "response"

    input_ids = torch.randint(0, cfg.vocab_size, (B, T))

    # Task-completion labels: -100 for instruction tokens (excluded from loss).
    labels = input_ids.clone()
    labels[:, :prefix_len] = HRMText.IGNORE_LABEL

    prefix_lens = torch.full((B,), prefix_len, dtype=torch.long)

    # ── Simulate training step with TBPTT warmup schedule ─────────────────────
    total_training_steps = 100000
    current_step = 0  # 20% of total → end of warmup period

    bp_steps = model.hrm.compute_bp_steps(current_step, total_training_steps)
    print(f"bp_steps at step {current_step}/{total_training_steps}: {bp_steps}")

    loss, logits = model(
        input_ids,
        labels=labels,
        prefix_lens=prefix_lens,
        bp_steps=bp_steps,
    )

    print(f"Loss:   {loss.item():.4f}")
    print(f"Logits: {logits.shape}")  # expected: [2, 16, 256]

    loss.backward()
    print("Backward pass OK.")
