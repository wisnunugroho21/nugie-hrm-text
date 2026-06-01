"""
Smoke test / quick sanity check for HRMText.

Runs a single forward + backward pass on randomly generated data to verify:
  - The model builds without errors.
  - Shapes are correct: logits should be [B, T, vocab_size].
  - The loss is a valid scalar and can be backpropagated.
  - The TBPTT warmup schedule produces a sensible bp_steps value.

This is intentionally tiny (vocab=256, hidden=64) so it runs in seconds on CPU.
"""

import torch
import torch.nn.functional as F

from hrm_text import HRMText

torch.manual_seed(42)

# Tokens with this label are excluded from the cross-entropy loss.
# PyTorch uses -100 as its default ignore index.
IGNORE_LABEL: int = -100

# ── Tiny model configuration for a quick smoke test ───────────────────────────
vocab_size = 256
hidden_size = 64
num_heads = 4
num_layers = 4      # → 2 layers per H/L module (each module gets half the layers)
ffn_hidden_size = 128
max_seq_len = 32
H_cycles = 2
L_cycles = 3

model = HRMText(
    vocab_size,
    hidden_size,
    max_seq_len,
    num_heads,
    num_kv_heads=2,
    H_layers=num_layers // 2,
    L_layers=num_layers // 2,
    H_cycles=H_cycles,
    L_cycles=L_cycles,
)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ── Dummy instruction–response batch ──────────────────────────────────────────
B, T = 2, 16
prefix_len = 6  # first 6 tokens are the "instruction"; the rest are the "response"

# Random token IDs for the entire batch.
input_ids = torch.randint(0, vocab_size, (B, T))

# Task-completion labels: instruction tokens get IGNORE_LABEL so the loss
# is computed only over the response tokens.
labels = input_ids.clone()
labels[:, :prefix_len] = IGNORE_LABEL

# prefix_lens tells the model where the instruction ends for each example.
prefix_lens = torch.full((B,), prefix_len, dtype=torch.long)

# ── Simulate a training step with the TBPTT warmup schedule ───────────────────
total_training_steps = 100_000
current_step = 0  # step 0 → start of training → smallest TBPTT window

# compute_bp_steps returns the appropriate TBPTT window for the current step.
bp_steps = model.hrm.compute_bp_steps(current_step, total_training_steps)
print(f"bp_steps at step {current_step}/{total_training_steps}: {bp_steps}")

# Forward pass: returns [B, T, vocab_size] logits.
logits = model(
    input_ids,
    prefix_lens=prefix_lens,
    bp_steps=bp_steps,
)

# Compute next-token prediction loss, skipping instruction tokens.
# Cast to float32 for numerical stability before cross-entropy.
loss = F.cross_entropy(
    logits.view(-1, vocab_size).float(),
    labels.view(-1).long(),
    ignore_index=IGNORE_LABEL,
)

print(f"Loss:   {loss.item():.4f}")
print(f"Logits: {logits.shape}")  # expected: [2, 16, 256]

# Backward pass to verify gradients flow through the TBPTT window.
loss.backward()
print("Backward pass OK.")
