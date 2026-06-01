import torch


def make_prefixlm_mask(
    prefix_lens: torch.Tensor, total_len: int, device: torch.device
) -> torch.Tensor:
    """
    Build a PrefixLM attention mask for instruction-response training.

    Standard causal (decoder-only) attention lets each token see only its
    past. PrefixLM modifies this so that instruction tokens (the "prefix")
    can also attend to *future* instruction tokens — giving the model
    encoder-like bidirectional context over the instruction before switching
    to causal generation for the response.

    Layout of a single example with prefix_len=3, total_len=6:
        Positions: [I0, I1, I2, R0, R1, R2]   (I = instruction, R = response)

        Allowed attention (True):
            I0 → I0, I1, I2          (bidirectional within prefix)
            I1 → I0, I1, I2
            I2 → I0, I1, I2
            R0 → I0, I1, I2, R0      (causal from here on)
            R1 → I0, I1, I2, R0, R1
            R2 → I0, I1, I2, R0, R1, R2

    Args:
        prefix_lens: [B] integer tensor — number of instruction tokens per example.
        total_len:   total sequence length T.
        device:      target device for the output mask.

    Returns:
        float_mask: [B, 1, T, T] additive float mask where
                      0.0  = token pair is allowed to attend,
                      -inf = token pair is blocked.
                    The shape [B, 1, T, T] broadcasts over attention heads.
    """
    B, T = prefix_lens.size(0), total_len

    # Start with a standard lower-triangular causal mask (True = can attend).
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    mask = causal.unsqueeze(0).expand(B, T, T).clone()  # [B, T, T]

    # For each example, allow full bidirectional attention within its prefix.
    for b in range(B):
        plen = int(prefix_lens[b].item())
        mask[b, :plen, :plen] = True  # all prefix tokens attend to each other

    # Convert the boolean mask to an additive float mask compatible with
    # PyTorch attention:  True → 0.0 (keep),  False → -inf (mask out).
    float_mask = torch.zeros(B, 1, T, T, dtype=torch.float32, device=device)
    float_mask.masked_fill_(~mask.unsqueeze(1), float("-inf"))
    return float_mask


def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    """
    In-place truncated-normal initialization.

    Draws values from N(0, 1), clamps them to ±3 standard deviations (to
    avoid extreme outlier weights), then rescales to the requested std.

    The magic constant 1.014_762... corrects for the slight variance reduction
    caused by clamping — it ensures the final distribution has the requested
    std to within a very small error.

    The @torch.no_grad() decorator on the caller side allows safe in-place
    use on nn.Parameters (which have requires_grad=True and would otherwise
    raise an error on in-place operations).
    """
    with torch.no_grad():
        return tensor.normal_().fmod_(3.0).mul_(1.014_762_601_732_121 * std)
