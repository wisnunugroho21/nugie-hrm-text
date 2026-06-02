"""
Adam-atan2 optimizer for PyTorch.

Replaces the standard Adam update:
    param -= lr * m_hat / (sqrt(v_hat) + eps)

with an atan2-based update that is epsilon-free and naturally bounded:
    param -= lr * (2/π) * atan2(m_hat, sqrt(v_hat))

The factor 2/π normalises the atan2 output from [-π/2, π/2] to [-1, 1],
giving similar step magnitudes to Adam without requiring a tuned epsilon.

Reference:
    Bernstein & Newhouse (2024) — "Old Optimizer, New Norm"
    https://arxiv.org/abs/2409.20325
"""

import math
import torch
from torch.optim import Optimizer


class AdamAtan2(Optimizer):
    """Adam optimizer with atan2-based update (no epsilon required).

    Args:
        params:      Iterable of parameters or parameter groups.
        lr:          Learning rate (default: 1e-3).
        betas:       Coefficients for computing running averages of gradient
                     and its square (default: (0.9, 0.999)).
        weight_decay: L2 penalty (decoupled, applied before the update).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
    ):
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimisation step.

        Args:
            closure: A closure that re-evaluates the model and returns the loss.

        Returns:
            loss (optional) — only if a closure is provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamAtan2 does not support sparse gradients.")

                state = self.state[p]

                # Initialise state on first step.
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                m, v = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # Decoupled weight decay (applied before parameter update).
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Update biased first and second moment estimates.
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias-correction factors.
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t

                # Bias-corrected moments.
                m_hat = m / bc1
                v_hat_sqrt = (v / bc2).sqrt()

                # atan2-based update: bounded in [-π/2, π/2], scaled to [-1, 1].
                update = torch.atan2(m_hat, v_hat_sqrt).mul_(2.0 / math.pi)

                p.add_(update, alpha=-lr)

        return loss
