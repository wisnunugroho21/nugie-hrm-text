import torch


def trunc_normal(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return tensor.normal_().fmod_(3.0).mul_(1.014_762_601_732_121 * std)


def make_prefixlm_mask(
    prefix_lens: torch.Tensor, total_len: int, device: torch.device
) -> torch.Tensor:
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


def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return tensor.normal_().fmod_(3.0).mul_(1.014_762_601_732_121 * std)
