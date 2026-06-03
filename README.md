# nugie-hrm-text

A minimal PyTorch implementation of **HRM-Text** — a Hierarchical Reasoning Model for language modelling that replaces the standard single-pass Transformer with a dual-timescale recurrent loop.

> Based on the paper: *"HRM-Text: Efficient Pretraining Beyond Scaling"* (2025)  
> Paper: <https://arxiv.org/abs/2605.20613> · Official code: <https://github.com/sapientinc/HRM-Text>

---

## What is HRM-Text?

A standard Transformer processes each token in a single forward pass. HRM-Text instead runs the model in a **nested recurrent loop**, similar to how the brain alternates between fast, local processing and slow, high-level planning.

### The Dual-Timescale Loop

Two modules alternate inside every forward pass:

| Module | Role | Update frequency |
|--------|------|-----------------|
| **H (High-level)** | Maintains broad semantic and planning context | Once per outer cycle |
| **L (Low-level)** | Performs fine-grained local refinement | `L_cycles` times per outer cycle |

With the defaults `H_cycles=2, L_cycles=3`, the loop runs: L L L H L L L H — 8 total module calls.

Both H and L share the same layer budget as a single-pass Transformer (each gets half the layers), so the total parameter count stays the same.

### Key Techniques

1. **MagicNorm** — Each recurrent module ends with a boundary RMSNorm. This keeps hidden-state variance bounded each step (like PostNorm) while preserving PreNorm's stable gradient flow through residual shortcuts.

2. **Truncated BPTT with warmup** — Only the last `K` recurrent steps are backpropagated. `K` starts small (2) and grows to a maximum (5) over the first 20% of training — a curriculum that stabilises early-stage gradients.

3. **PrefixLM masking** — Instruction tokens attend to each other bidirectionally; response tokens use standard left-to-right causal masking. This gives the model encoder-like context over instructions.

4. **Task-completion objective** — Loss is computed only on response tokens; instruction tokens receive label `-100` (PyTorch's ignore index).

---

## Project Structure

```
nugie-hrm-text/
├── main.py             # Smoke test: forward + backward pass
├── hrm_text.py         # Top-level model: embedding → HRM → LM head
├── hrm.py              # HierarchicalReasoningModel: the dual-timescale loop
├── reasoning_module.py # Single recurrent module (H or L) with MagicNorm
├── transformer.py      # TransformerBlock: PreNorm attention + FFN
├── attention.py        # SigmoidGatedAttention with GQA and RoPE
├── swiglu.py           # SwiGLU feed-forward network
├── rope.py             # Rotary Positional Embedding (RoPE)
├── rmsnorm.py          # Parameterless RMSNorm
├── utilities.py        # PrefixLM mask builder + truncated-normal init
└── v1.py               # Self-contained single-file reference implementation
```

### Module Relationships

```
HRMText (hrm_text.py)
└── HierarchicalReasoningModel (hrm.py)
    ├── L_net: ReasoningModule (reasoning_module.py)
    │   └── [N × TransformerBlock (transformer.py)]
    │           ├── SigmoidGatedAttention (attention.py)
    │           │   └── RotaryPositionalEmbedding (rope.py)
    │           └── SwiGLUFeedForward (swiglu.py)
    └── H_net: ReasoningModule (same structure as L_net)
```

---

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.1+ (uses `nn.Buffer` and `torch.set_grad_enabled`)

### Run the Smoke Test

```bash
python main.py
```

Expected output (values may vary slightly):

```
Model parameters: 131,904
bp_steps at step 0/100000: 2
Loss:   5.5432
Logits: torch.Size([2, 16, 256])
Backward pass OK.
```

### Using the Model in Your Code

```python
import torch
from hrm_text import HRMText

model = HRMText(
    vocab_size=32_000,
    hidden_size=512,
    seq_len=2048,
    num_heads=8,
    num_kv_heads=4,   # GQA: 2 Q heads per KV head
    H_layers=4,
    L_layers=4,
    H_cycles=2,
    L_cycles=3,
)

# --- Training step ---
input_ids  = torch.randint(0, 32_000, (batch_size, seq_len))
prefix_lens = torch.tensor([prefix_length] * batch_size)  # instruction lengths

bp_steps = model.hrm.compute_bp_steps(current_step, total_steps)

logits = model(input_ids, prefix_lens=prefix_lens, bp_steps=bp_steps)
# logits: [batch_size, seq_len, vocab_size]

# --- Inference (no TBPTT needed) ---
logits = model(input_ids)
```

---

## Architecture Details

### Attention (`attention.py`)

**Grouped Query Attention (GQA)** reduces the number of K/V heads relative to Q heads (`num_kv_heads < num_heads`), cutting memory and compute without sacrificing expressivity.

A **sigmoid gate** is applied to the attention output:
```
out = sigmoid(gate) × attn(Q, K, V)
```
This data-dependent gate lets the model suppress irrelevant context, similar to LSTM/GRU gating.

### Feed-Forward Network (`swiglu.py`)

Uses **SwiGLU**: projects to a gate and value branch, applies `SiLU(gate) × value`, then projects back. More expressive than plain ReLU/GELU at the same parameter budget.

### Positional Encoding (`rope.py`)

**Rotary Position Embeddings (RoPE)** rotate Q and K vectors by position-dependent angles so the attention dot product directly encodes relative distance. No learned position table is needed.

### Normalization (`rmsnorm.py`)

**Parameterless RMSNorm** — no learnable scale (γ) or shift (β). Divides activations by their root mean square to control variance without extra parameters.

### Initialization (`utilities.py`)

**Truncated normal** (`trunc_normal`): draws from N(0,1), clamps to ±3σ, then rescales. This avoids extreme outlier weights at initialization without changing the target distribution significantly.

---

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | — | Number of tokens in the vocabulary |
| `hidden_size` | — | Width of all hidden states (D) |
| `seq_len` | — | Maximum sequence length |
| `num_heads` | 4 | Number of attention heads |
| `num_kv_heads` | 2 | Number of KV heads for GQA |
| `H_layers` | 2 | Transformer layers in the H module |
| `L_layers` | 2 | Transformer layers in the L module |
| `H_cycles` | 3 | Number of outer H cycles |
| `L_cycles` | 3 | Number of L steps per H cycle |
| `bp_warmup_ratio` | 0.2 | Fraction of training for TBPTT warmup |
| `bp_min_steps` | 2 | TBPTT window at the start of training |
| `bp_max_steps` | 5 | TBPTT window after warmup |

---

## Reference

```
@article{hrm-text-2025,
  title  = {HRM-Text: Efficient Pretraining Beyond Scaling},
  year   = {2025},
  url    = {https://arxiv.org/abs/2605.20613}
}
```
