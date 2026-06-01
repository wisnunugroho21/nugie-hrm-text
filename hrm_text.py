import math

import torch
import torch.nn as nn

from hrm import HierarchicalReasoningModel
from utilities import make_prefixlm_mask, trunc_normal_


class HRMText(nn.Module):
    """
    Full HRM-Text model: token embedding → HRM core → LM head.

    This wraps HierarchicalReasoningModel with the input and output layers
    needed for language modelling:

      1. Token embedding — maps integer token IDs to dense vectors.
         Embeddings are scaled by √D (the inverse of the LeCun init std) to
         keep activation norms stable throughout training.

      2. HRM core — the dual-timescale recurrent backbone (see hrm.py).

      3. LM head — a linear projection from hidden_size back to vocab_size,
         producing unnormalized logits for next-token prediction.

    Attention masking:
      If prefix_lens is provided, a PrefixLM mask is built:
        - Instruction tokens (the "prefix") attend to each other bidirectionally.
        - Response tokens use standard left-to-right causal masking.
      Without prefix_lens, standard causal masking is used throughout.

    Training:
      During training, pass bp_steps = model.hrm.compute_bp_steps(step, total_steps)
      to apply the Truncated BPTT warmup schedule. During inference, any
      bp_steps value is fine (or omit it to use the default of 5).
    """

    def __init__(
        self,
        vocab_size: int,
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
        bp_warmup_ratio: float = 0.2,  # fraction of training steps for TBPTT warmup
        bp_min_steps: int = 2,         # TBPTT window at the start of training
        bp_max_steps: int = 5,         # TBPTT window at the end of warmup
    ):
        super().__init__()

        # Token embedding with LeCun-style initialization (std = 1/√D).
        # The scale multiplier (√D) is applied at runtime so the effective
        # embedding norm is ~1.0 regardless of hidden_size.
        init_std = 1.0 / math.sqrt(hidden_size)  # LeCun std = 1/√D
        self.embed_scale = 1.0 / init_std         # runtime multiplier = √D
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        trunc_normal_(self.embed_tokens.weight, std=init_std)

        # HRM recurrent core — handles all the dual-timescale computation.
        self.hrm = HierarchicalReasoningModel(
            hidden_size,
            seq_len,
            num_heads,
            num_kv_heads,
            H_layers,
            L_layers,
            H_cycles,
            L_cycles,
            norm_eps,
            expansion,
            bp_warmup_ratio,
            bp_min_steps,
            bp_max_steps,
        )

        # Language model head: projects each position's hidden state to logits
        # over the full vocabulary. No bias — consistent with weight tying style.
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        trunc_normal_(self.lm_head.weight, std=init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        prefix_lens: torch.Tensor | None = None,
        bp_steps: int = 5,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:   [B, T]  — integer token IDs for a batch of sequences.
            prefix_lens: [B]     — number of instruction tokens per example.
                                   When given, PrefixLM attention is applied;
                                   otherwise, standard causal masking is used.
            bp_steps:    int     — TBPTT window size. During training use
                                   hrm.compute_bp_steps(step, total_steps).

        Returns:
            logits: [B, T, vocab_size] — unnormalized next-token scores.
        """
        B, T = input_ids.shape

        # Step 1: Embed tokens and scale to unit norm territory.
        x = self.embed_scale * self.embed_tokens(input_ids)  # [B, T, D]

        # Step 2: Build the attention mask.
        if prefix_lens is not None:
            # PrefixLM mask: bidirectional over instruction tokens,
            # causal over response tokens (see utilities.make_prefixlm_mask).
            attn_mask = make_prefixlm_mask(prefix_lens, T, input_ids.device)
        else:
            # No mask provided — attention.py applies standard causal masking.
            attn_mask = None

        # Step 3: Run the HRM recurrent forward pass.
        z_H = self.hrm(x, attn_mask=attn_mask, bp_steps=bp_steps)  # [B, T, D]

        # Step 4: Project hidden states to vocabulary logits.
        return self.lm_head(z_H)  # [B, T, vocab_size]

