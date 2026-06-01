import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RotaryPositionalEmbedding


class SigmoidGatedAttention(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, num_kv_heads: int, max_seq_len: int
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, (
            "hidden_size must be divisible by num_heads"
        )
        assert num_heads % num_kv_heads == 0, (
            "num_heads must be divisible by num_kv_heads"
        )

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads  # Q heads per KV head
        self.head_dim = hidden_size // num_heads

        # Q and gate are projected to full num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)

        # K and V are projected to the smaller num_kv_heads * head_dim (GQA)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

        self.scale = self.head_dim**-0.5

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x:         [B, S, D]
        attn_mask: [B, 1, S, S] additive float mask  (0.0 = attend, -inf = block)
                   or None → standard causal masking is applied.
        """
        B, S, D = x.shape
        H, Hkv, G, Dh = (
            self.num_heads,
            self.num_kv_heads,
            self.num_groups,
            self.head_dim,
        )

        # --- Project Q / K / V / gate ----------------------------------------
        q = self.q_proj(x)     # (B, S, H*Dh)
        k = self.k_proj(x)     # (B, S, Hkv*Dh)
        v = self.v_proj(x)     # (B, S, Hkv*Dh)
        g = self.gate_proj(x)  # (B, S, H*Dh)

        # --- Reshape: (B, S, H*Dh) → (B, S, H, Dh) --------------------------
        q = q.view(B, S, H, Dh)
        k = k.view(B, S, Hkv, Dh)
        v = v.view(B, S, Hkv, Dh)
        g = g.view(B, S, H, Dh)

        # ── Apply RoPE to Q and K ─────────────────────────────────────────────
        q = self.rope(q)
        k = self.rope(k)

        # --- Transpose to [B, n_heads, S, head_dim] for SDPA ------------------
        q = q.transpose(1, 2)  # (B, H,   S, Dh)
        k = k.transpose(1, 2)  # (B, Hkv, S, Dh)
        v = v.transpose(1, 2)  # (B, Hkv, S, Dh)

        # --- Expand KV heads to match Q heads (GQA) ---------------------------
        k = k.repeat_interleave(G, dim=1)  # (B, H, S, Dh)
        v = v.repeat_interleave(G, dim=1)  # (B, H, S, Dh)

        # --- Scaled dot-product attention -------------------------------------
        # scores[b, h, i, j] = (Q[b,h,i,:] · K[b,h,j,:]) / √Dh
        scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, S, S)

        if attn_mask is not None:
            # attn_mask is an additive float mask: 0.0 = attend, -inf = block.
            scores = scores + attn_mask
        else:
            # No custom mask → apply a standard causal (lower-triangular) mask
            # so each token only attends to itself and earlier positions.
            causal_mask = torch.ones(S, S, dtype=torch.bool, device=q.device).tril()
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # (B, H, S, S)
        attn_out = attn_weights @ v               # (B, H, S, Dh)

        # --- Merge heads: (B, H, S, Dh) → (B, S, H, Dh) → (B, S, D) ---------
        attn_out = attn_out.transpose(1, 2)  # (B, S, H, Dh)

        # ── Sigmoid gate: element-wise gating of attention output ─────────────
        out = torch.sigmoid(g) * attn_out
        out = out.reshape(B, S, D)

        return self.out_proj(out)
