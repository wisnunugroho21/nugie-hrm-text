import torch

from hrm_text import IGNORE_LABEL, HRMText

torch.manual_seed(42)

# Tiny config for a quick forward/backward smoke test.
vocab_size=256
hidden_size=64
num_heads=4
num_layers=4  # → 2 layers per H/L module (half_layers=True)
ffn_hidden_size=128
max_seq_len=32
H_cycles=2
L_cycles=3

model = HRMText(vocab_size, hidden_size, max_seq_len, num_heads, num_kv_heads=2, H_layers=num_layers//2, L_layers=num_layers//2, H_cycles=H_cycles, L_cycles=L_cycles)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ── Dummy instruction–response batch ──────────────────────────────────────
B, T = 2, 16
prefix_len = 6  # first 6 tokens = "instruction"; rest = "response"

input_ids = torch.randint(0, vocab_size, (B, T))

# Task-completion labels: -100 for instruction tokens (excluded from loss).
labels = input_ids.clone()
labels[:, :prefix_len] = IGNORE_LABEL

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