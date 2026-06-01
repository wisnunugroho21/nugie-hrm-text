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

        # Q is projected to full num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)

        # K and V are projected to the smaller num_kv_heads * head_dim
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

        self.scale = self.head_dim**-0.5

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, S, D = x.shape
        H, Hkv, G, Dh = (
            self.num_heads,
            self.num_kv_heads,
            self.num_groups,
            self.head_dim,
        )

        # --- Project G / Q / K / V -----------------------------------------------
        g = self.gate_proj(x)  # (B, H,   S, Dh)
        q = self.q_proj(x)  # (B, H,   S, Dh)
        k = self.k_proj(x)  # (B, Hkv, S, Dh)
        v = self.v_proj(x)  # (B, Hkv, S, Dh)

        # --- Reshape: (B, S, D) → (B, H, S, Dh) ------------------------------
        g = g.view(B, S, H, Dh)  # (B, H,   S, Dh)
        q = q.view(B, S, H, Dh)  # (B, H,   S, Dh)
        k = k.view(B, S, Hkv, Dh)  # (B, Hkv, S, Dh)
        v = v.view(B, S, Hkv, Dh)  # (B, Hkv, S, Dh)

        # ── Apply RoPE to Q and K ────────────────────────────────────────────
        q = self.rope(q)
        k = self.rope(k)

        # --- Transpose to [B, n_heads, T, head_dim] --------------------------
        q = q.transpose(1, 2)  # (B, H,   S, Dh)
        k = k.transpose(1, 2)  # (B, Hkv, S, Dh)
        v = v.transpose(1, 2)  # (B, Hkv, S, Dh)

        # --- Each KV head is shared by G query heads — expand to match Q -----
        k = k.repeat_interleave(G, dim=1)  # (B, H, S, Dh)
        v = v.repeat_interleave(G, dim=1)  # (B, H, S, Dh)

        # --- Scaled dot-product attention ------------------------------------
        # Note: Using math formula for clarity. In production, use F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, S, S)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = attn_weights @ v

        # --- Aggregate values, merge heads, and project ----------------------
        attn_out = attn_out.transpose(1, 2)  # (B, T, H, Dh)

        # ── Sigmoid gate ──────────────────────────────────────────────────────
        out = torch.sigmoid(g) * attn_out  # element-wise gate
        out = out.reshape(B, S, D)

        return self.out_proj(out)
