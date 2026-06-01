import math

import torch
import torch.nn as nn

from hrm import HierarchicalReasoningModel
from utilities import make_prefixlm_mask


class HRMText(nn.Module):
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

        # ── Token embedding (scaled) ──────────────────────────────────────────
        init_std = 1.0 / math.sqrt(hidden_size)  # LeCun std = 1/√D
        self.embed_scale = 1.0 / init_std  # runtime multiplier (= √D)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # ── HRM core ──────────────────────────────────────────────────────────
        self.hrm = HierarchicalReasoningModel(
            vocab_size,
            hidden_size,
            seq_len,
            num_heads,
            num_kv_heads,
            H_layers,
            L_layers,
            H_cycles,  # high-level cycles
            L_cycles,  # low-level steps per cycle
            norm_eps,
            expansion,
            bp_warmup_ratio,  # Fraction of total steps for the warmup phase
            bp_min_steps,  # TBPTT steps at the start of training  (K = 2)
            bp_max_steps,  # TBPTT steps at the end of warmup      (K = 5)
        )

        # ── Output projection (no bias) ────────────────────────────────────────
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        prefix_lens: torch.Tensor | None = None,
        bp_steps: int = 5,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
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

        return logits
