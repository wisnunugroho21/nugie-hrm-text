"""
pretraining.py — HRM-Text pre-training loop.

Paper : "HRM-Text: Efficient Pretraining Beyond Scaling"
         https://arxiv.org/abs/2605.20613
Code  : https://github.com/sapientinc/HRM-Text

────────────────────────────────────────────────────────────────────────────────
Training Objectives (Section 2.2 of the paper)
────────────────────────────────────────────────────────────────────────────────

1. Task-completion objective
     Train on instruction–response pairs. Loss is computed *only* over
     response tokens; instruction tokens receive label = IGNORE_LABEL (-100)
     and are excluded from the cross-entropy.

2. PrefixLM attention mask
     Instruction tokens attend to each other bidirectionally (encoder-like).
     Response tokens use standard left-to-right causal masking.
     The mask is built automatically inside HRMText when `prefix_lens` is given.

3. Warmup deep credit assignment — Truncated BPTT (Section 2.1.2)
     Back-propagation is truncated to the last K recurrent steps so that
     gradients remain stable. K starts at `bp_min_steps` and grows linearly
     to `bp_max_steps` over the first `bp_warmup_ratio` fraction of training.
     Earlier steps run under torch.no_grad() and are excluded from the graph.

Learning-rate schedule
     Linear warmup from 0 → peak LR over `lr_warmup_steps`, then cosine
     decay from peak LR down to `lr * lr_min_ratio`.
"""

import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hrm_text import HRMText

# Tokens with this label are skipped by PyTorch's cross-entropy loss.
IGNORE_LABEL: int = -100


# ──────────────────────────────────────────────────────────────────────────────
# 1. Training Configuration
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PretrainConfig:
    """
    All hyper-parameters for one HRM-Text pre-training run.

    Architecture fields map 1-to-1 with HRMText.__init__ keyword arguments.
    Adjust them to match your tokenizer vocabulary and desired model size.

    The official paper uses AdamATan2 (a scale-invariant Adam variant).
    This implementation uses standard AdamW for broad compatibility; the
    learning-rate schedule and TBPTT warmup are identical to the paper.
    """

    # ── Dataset ───────────────────────────────────────────────────────────────
    max_seq_len: int = 2048  # max tokens per example (instruction + response)

    # ── Model architecture ────────────────────────────────────────────────────
    vocab_size: int = 32_000
    hidden_size: int = 1024
    num_heads: int = 8
    num_kv_heads: int = 4  # GQA: fewer KV heads share Q heads
    H_layers: int = 4  # transformer layers in the slow H module
    L_layers: int = 4  # transformer layers in the fast L module
    H_cycles: int = 2  # outer H cycles per forward pass
    L_cycles: int = 3  # inner L cycles per H cycle (H2L3 in the paper)
    norm_eps: float = 1e-6
    expansion: float = 4 / 3  # SwiGLU inner-width multiplier

    # ── Training ──────────────────────────────────────────────────────────────
    global_batch_size: int = 512  # sequences per step (all GPUs combined)
    epochs: int = 4  # passes over the training data
    seed: int = 0

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr: float = 3e-4  # peak learning rate
    lr_min_ratio: float = 0.1  # final LR = lr * lr_min_ratio
    lr_warmup_steps: int = 2000  # steps to linearly ramp LR from 0 to lr
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # gradient norm clipping (0 = disabled)

    # ── EMA (Exponential Moving Average of weights) ────────────────────────────
    ema_decay: Optional[float] = 0.9999  # set to None to disable EMA

    # ── Truncated BPTT schedule ───────────────────────────────────────────────
    bp_warmup_ratio: float = 0.2  # fraction of total steps to warm up bp_steps
    bp_min_steps: int = 2  # TBPTT window at the very start of training
    bp_max_steps: int = 5  # TBPTT window after warmup is complete

    # ── Checkpointing & logging ────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1  # save a checkpoint every N epochs
    log_interval: int = 10  # print a metrics line every N steps

    # ── Collation ─────────────────────────────────────────────────────────────
    pad_token_id: int = 0  # used to right-pad variable-length sequences

    # ── Device ────────────────────────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Dataset — instruction–response pairs
# ──────────────────────────────────────────────────────────────────────────────


class InstructionDataset(Dataset):
    """
    Stores tokenized instruction–response pairs for the task-completion objective.

    Each example is represented as:
      • input_ids  : [T] int64 — full concatenated sequence (instruction + response).
      • prefix_len : int       — number of instruction tokens.

    At collation time (see collate_fn), labels are derived from input_ids by
    masking the instruction positions with IGNORE_LABEL so only the response
    tokens contribute to the cross-entropy loss.

    Expected input format:
        A list of (instruction_ids, response_ids) pairs where each element is
        a plain list of integer token IDs.
    """

    def __init__(
        self,
        samples: list[tuple[list[int], list[int]]],
        max_seq_len: int,
    ):
        self.examples = []
        for instr, resp in samples:
            # Concatenate instruction and response, then right-truncate.
            seq = (instr + resp)[:max_seq_len]
            self.examples.append(
                {
                    "input_ids": torch.tensor(seq, dtype=torch.long),
                    "prefix_len": min(len(instr), max_seq_len),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


# ──────────────────────────────────────────────────────────────────────────────
# 3. HuggingFace Dataset — official HRM-Text pretraining data
# ──────────────────────────────────────────────────────────────────────────────


class HRMTextDataset(Dataset):
    """
    Official HRM-Text pretraining dataset loaded from HuggingFace.

    Source   : sapientinc/HRM-Text-data-io-cleaned-20260515
    Tokenizer: sapientinc/HRM-Text-1B  (65 536-token custom BPE)

    Each row in the raw dataset has three fields:
        instruction : str  — the question or problem statement
        response    : str  — the expected answer
        condition   : str  — task-style tag (e.g. "direct", "cot", "synth,cot")

    These are encoded into a single token sequence using the format from the paper:

        <|im_start|> {condition_tokens...} {instruction_tokens} <|im_end|>
        {response_tokens} <eos>
                          ↑
             prefix boundary (prefix_len)

    The prefix (everything up to and including <|im_end|>) receives bidirectional
    PrefixLM attention. Response tokens are generated causally with full supervision.

    Requirements
    ────────────
        pip install datasets transformers
    """

    # Condition tag → special token string (from the model card).
    # A condition string may be comma-separated (e.g. "synth,cot") to stack tokens.
    # These special tokens signal the expected response style to the model:
    #   direct  → straightforward answer, no chain-of-thought
    #   cot     → chain-of-thought reasoning
    #   noisy   → noisy / web-scraped training signal
    #   synth   → synthetic / curated data
    CONDITION_MAP: dict[str, str] = {
        "direct": "<|object_ref_start|>",
        "cot": "<|object_ref_end|>",
        "noisy": "<|quad_start|>",
        "synth": "<|quad_end|>",
    }

    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_seq_len: int = 2048,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            tokenizer   : HuggingFace tokenizer from sapientinc/HRM-Text-1B.
            split       : dataset split ("train" is the only available split).
            max_seq_len : sequences longer than this are right-truncated.
            max_samples : if set, load only the first N examples; useful for
                          quick experiments. Leave as None for the full dataset.
        """
        from datasets import load_dataset

        raw = load_dataset(
            "sapientinc/HRM-Text-data-io-cleaned-20260515",
            split=split,
        )
        if max_samples is not None:
            raw = raw.select(range(min(max_samples, len(raw))))

        self.examples: list[dict] = []

        # Pre-compute the bracket token IDs once so we don't re-encode them per row.
        im_start_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
        im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

        print(f"[HRMTextDataset] Tokenizing {len(raw):,} examples …")
        for row in raw:
            # Build the instruction prefix: <|im_start|> condition instruction <|im_end|>
            condition_ids: list[int] = []
            for tag in row.get("condition", "direct").split(","):
                tag = tag.strip()
                special = self.CONDITION_MAP.get(tag)
                if special is not None:
                    condition_ids += tokenizer.encode(special, add_special_tokens=False)

            instr_ids = (
                im_start_ids
                + condition_ids
                + tokenizer.encode(row["instruction"], add_special_tokens=False)
                + im_end_ids
            )
            resp_ids = (
                tokenizer.encode(row["response"], add_special_tokens=False) + eos_ids
            )

            # Right-truncate the full sequence to max_seq_len.
            seq = (instr_ids + resp_ids)[:max_seq_len]
            plen = min(len(instr_ids), max_seq_len)

            self.examples.append(
                {
                    "input_ids": torch.tensor(seq, dtype=torch.long),
                    "prefix_len": plen,
                }
            )

        print(f"[HRMTextDataset] Ready — {len(self.examples):,} examples.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def collate_fn(
    batch: list[dict],
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    """
    Pad a variable-length batch and build per-token supervision labels.

    Padding is appended on the right to match the longest sequence. Both
    padded positions and instruction positions receive IGNORE_LABEL so
    the loss covers only the response tokens.

    Returns:
        input_ids   : [B, T]  — padded token IDs (model input).
        labels      : [B, T]  — IGNORE_LABEL everywhere except response tokens.
        prefix_lens : [B]     — instruction lengths, for the PrefixLM mask.
    """
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids_list, labels_list, prefix_lens = [], [], []
    for item in batch:
        ids = item["input_ids"]  # [T]
        plen = item["prefix_len"]
        T = ids.size(0)
        pad = max_len - T

        # Right-pad input_ids to the batch's max length.
        padded = F.pad(ids, (0, pad), value=pad_token_id)  # [max_len]

        # Build labels:
        #   • instruction tokens (0 … plen-1) → IGNORE_LABEL (no loss)
        #   • response    tokens (plen … T-1) → actual token IDs (supervised)
        #   • padding     tokens (T … max-1)  → IGNORE_LABEL (excluded)
        lbl = padded.clone()
        lbl[:plen] = IGNORE_LABEL
        if pad > 0:
            lbl[T:] = IGNORE_LABEL

        input_ids_list.append(padded)
        labels_list.append(lbl)
        prefix_lens.append(plen)

    return {
        "input_ids": torch.stack(input_ids_list),  # [B, T]
        "labels": torch.stack(labels_list),  # [B, T]
        "prefix_lens": torch.tensor(prefix_lens, dtype=torch.long),  # [B]
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. EMA — Exponential Moving Average of model weights
# ──────────────────────────────────────────────────────────────────────────────


class EMA:
    """
    Tracks an exponential moving average of all trainable model parameters.

    After every optimizer step, call ema.update() to blend the live weights
    into the shadow copy:

        shadow = decay * shadow + (1 - decay) * live_weight

    EMA weights are smoother than the live weights and are conventionally
    preferred for evaluation and checkpoint export. The official HRM-Text code
    integrates EMA directly into its AdamATan2 optimizer; this class provides
    the same behaviour alongside standard AdamW.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        # Shadow copy — detached from the computation graph at all times.
        self.shadow: dict[str, torch.Tensor] = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self):
        """Blend the current model weights into the shadow copy."""
        for name, p in self.model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    p.detach(), alpha=1.0 - self.decay
                )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]):
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Learning-Rate Schedule
# ──────────────────────────────────────────────────────────────────────────────


def compute_lr(step: int, total_steps: int, config: PretrainConfig) -> float:
    """
    Linear-warmup → cosine-decay learning-rate schedule.

      • Steps 0 … lr_warmup_steps-1 : LR ramps linearly from 0 to config.lr.
      • Steps lr_warmup_steps … end  : cosine decay from config.lr
                                       down to config.lr * config.lr_min_ratio.

    The cosine factor reaches 1.0 at the start of decay and 0.0 at the end,
    so the LR floor is exactly lr_min_ratio * lr.
    """
    if step < config.lr_warmup_steps:
        return config.lr * max(step, 1) / max(config.lr_warmup_steps, 1)

    # Progress within the decay phase: 0.0 at warmup end, 1.0 at training end.
    progress = (step - config.lr_warmup_steps) / max(
        total_steps - config.lr_warmup_steps, 1
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.lr * (config.lr_min_ratio + (1.0 - config.lr_min_ratio) * cosine)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Checkpointing
# ──────────────────────────────────────────────────────────────────────────────


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: Optional[EMA],
    step: int,
    epoch: int,
    config: PretrainConfig,
):
    """Save model weights, optimizer state, and EMA shadow weights."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "config": config,
        },
        path,
    )
    print(f"[Checkpoint] Saved  → {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: Optional[EMA],
    device: str,
) -> tuple[int, int]:
    """
    Restore model, optimizer, and EMA from a saved checkpoint.

    Returns:
        (step, epoch) — training cursor so the caller can resume correctly.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if ema is not None and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])
    step, epoch = ckpt["step"], ckpt["epoch"]
    print(f"[Checkpoint] Loaded ← {path}  (epoch={epoch}, step={step})")
    return step, epoch


# ──────────────────────────────────────────────────────────────────────────────
# 6. Training Loop
# ──────────────────────────────────────────────────────────────────────────────


def train(
    config: PretrainConfig,
    dataset: Dataset,
    resume_from: Optional[str] = None,
) -> tuple[HRMText, Optional[EMA]]:
    """
    Run HRM-Text pre-training.

    Per-step algorithm
    ──────────────────
      1. Compute bp_steps via the TBPTT warmup schedule.
      2. Forward pass — HRMText builds the PrefixLM mask from prefix_lens,
         giving bidirectional attention over instruction tokens and causal
         attention over response tokens.
      3. Cross-entropy loss over response tokens only (task-completion).
      4. Backward pass, optional gradient clipping, optimizer step.
      5. EMA shadow update.
      6. Learning-rate update (warmup + cosine decay).

    Args:
        config      : all training hyperparameters (see PretrainConfig).
        dataset     : a torch Dataset yielding {"input_ids", "prefix_len"} dicts.
                      Use HRMTextDataset for the official pretraining data, or
                      InstructionDataset for pre-tokenized custom data.
        resume_from : optional path to a .pt checkpoint file to resume from.

    Returns:
        (model, ema) — the trained model and its EMA, or (model, None) if EMA
        is disabled.
    """
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    # ── Dataloader ────────────────────────────────────────────────────────────
    dataloader = DataLoader(
        dataset,
        batch_size=config.global_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=config.pad_token_id),
        drop_last=True,  # drop partial batches to keep gradient norms stable
        num_workers=0,
    )

    total_steps = config.epochs * len(dataloader)

    # ── Model ─────────────────────────────────────────────────────────────────
    # HRMText wraps the HRM recurrent core with token embeddings and an LM head.
    # The bp_* parameters control the TBPTT warmup schedule inside the HRM.
    model = HRMText(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        seq_len=config.max_seq_len,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        H_layers=config.H_layers,
        L_layers=config.L_layers,
        H_cycles=config.H_cycles,
        L_cycles=config.L_cycles,
        norm_eps=config.norm_eps,
        expansion=config.expansion,
        bp_warmup_ratio=config.bp_warmup_ratio,
        bp_min_steps=config.bp_min_steps,
        bp_max_steps=config.bp_max_steps,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {num_params:,}")
    print(
        f"Total steps: {total_steps}  "
        f"({config.epochs} epochs × {len(dataloader)} steps/epoch)"
    )

    # ── Optimizer — apply weight decay only to weight matrices, not to biases
    #    or 1-D tensors (embeddings excluded here have shape [V, D] so ndim==2
    #    and DO get weight decay, which matches the official implementation).
    decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.lr,
        betas=(config.beta1, config.beta2),
    )

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema = EMA(model, decay=config.ema_decay) if config.ema_decay is not None else None

    # ── Resume from a previous checkpoint ────────────────────────────────────
    start_step, start_epoch = 0, 1
    if resume_from is not None:
        start_step, start_epoch = load_checkpoint(
            resume_from, model, optimizer, ema, config.device
        )
        start_epoch += 1  # the loaded epoch is complete; begin the next one

    # ── Main training loop ────────────────────────────────────────────────────
    step = start_step
    model.train()

    for epoch in range(start_epoch, config.epochs + 1):
        epoch_loss_sum = 0.0
        epoch_token_count = 0

        for batch in dataloader:
            step += 1

            input_ids = batch["input_ids"].to(device)  # [B, T]
            labels = batch["labels"].to(device)  # [B, T]
            prefix_lens = batch["prefix_lens"].to(device)  # [B]

            # ── 1. Compute TBPTT window for this step ─────────────────────
            # K starts small (bp_min_steps) and linearly grows to bp_max_steps
            # over the first bp_warmup_ratio of training, acting as a
            # curriculum: short gradients early in training are more stable
            # when recurrent states are noisy.
            bp_steps = model.hrm.compute_bp_steps(step, total_steps)

            # ── 2. Forward pass ───────────────────────────────────────────
            # HRMText builds the PrefixLM mask when prefix_lens is provided:
            #   - instruction positions → bidirectional attention
            #   - response positions    → left-to-right causal attention
            logits = model(input_ids, prefix_lens=prefix_lens, bp_steps=bp_steps)
            # logits: [B, T, vocab_size]

            # ── 3. Task-completion loss (response tokens only) ────────────
            # Instruction tokens and padding carry IGNORE_LABEL and are
            # excluded from the mean by F.cross_entropy's ignore_index.
            # Cast logits to float32 for numerical stability before softmax.
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size).float(),  # [B*T, vocab_size]
                labels.view(-1),  # [B*T]
                ignore_index=IGNORE_LABEL,
            )

            # ── 4. Backward + gradient clipping + optimizer step ──────────
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            # ── 5. EMA update ──────────────────────────────────────────────
            if ema is not None:
                ema.update()

            # ── 6. Learning-rate update ────────────────────────────────────
            lr = compute_lr(step, total_steps, config)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Track loss weighted by supervised token count for a proper average.
            n_supervised = (labels != IGNORE_LABEL).sum().item()
            epoch_loss_sum += loss.item() * n_supervised
            epoch_token_count += n_supervised

            if step % config.log_interval == 0:
                print(
                    f"epoch {epoch:3d} | step {step:6d}/{total_steps} "
                    f"| loss {loss.item():.4f} | lr {lr:.2e} | bp_steps {bp_steps}"
                )

        # ── End-of-epoch summary ──────────────────────────────────────────────
        avg_loss = epoch_loss_sum / max(epoch_token_count, 1)
        print(f"── epoch {epoch} done | avg loss {avg_loss:.4f}")

        # Save a checkpoint at the requested interval and always on the last epoch.
        if epoch % config.checkpoint_interval == 0 or epoch == config.epochs:
            ckpt_path = os.path.join(config.checkpoint_dir, f"epoch_{epoch:04d}.pt")
            save_checkpoint(ckpt_path, model, optimizer, ema, step, epoch, config)

    print("Pre-training complete.")
    return model, ema


# ──────────────────────────────────────────────────────────────────────────────
# 7. Entry Point — smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Pre-train an HRM-Text model on the official HRM-Text pretraining data.

    The tokenizer and dataset are both pulled from HuggingFace:
        tokenizer : sapientinc/HRM-Text-1B
        dataset   : sapientinc/HRM-Text-data-io-cleaned-20260515

    Architecture and optimizer settings below match the 1B model from the paper
    (Section 3.1). Reduce hidden_size / num_heads / layers for smaller runs.
    Set max_samples=None to use the full dataset for production training.

    Requirements:
        pip install datasets transformers
    """
    from transformers import AutoTokenizer

    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained("sapientinc/HRM-Text-1B")

    # Use the tokenizer's pad token; fall back to EOS if none is explicitly defined.
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    cfg = PretrainConfig(
        # ── Architecture: 1B-model dimensions (Section 3.1) ───────────────────
        vocab_size=tokenizer.vocab_size,  # 65 536
        hidden_size=1536,
        num_heads=12,
        num_kv_heads=6,
        H_layers=16,
        L_layers=16,
        H_cycles=2,
        L_cycles=3,
        max_seq_len=4096,
        pad_token_id=pad_id,
        # ── Optimizer: paper values (Section 2.2) ─────────────────────────────
        global_batch_size=4,  # increase for multi-GPU / real runs
        epochs=1,
        lr=2.2e-4,
        lr_warmup_steps=2000,
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.1,
        ema_decay=0.9999,
        checkpoint_dir="checkpoints",
        log_interval=10,
    )

    dataset = HRMTextDataset(
        tokenizer=tokenizer,
        split="train",
        max_seq_len=cfg.max_seq_len,
        max_samples=64,  # set to None to use the full dataset
    )

    train(cfg, dataset)
