from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch

#calculate lr
def get_lr_cosine(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    # the first phase: smaller than warmup_iters
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    # the second phase: larger than warmup_iters smaller than cosine_cycle_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    else:
        decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)
    

# AdamW optimizer
class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                state = self.state[p]
                alpha = group["lr"]
                beta_1, beta_2 = group["betas"]
                eps = group["eps"]
                t = state.get("t", 1)
                prev_m_t = state.get("m", torch.zeros_like(grad))
                prev_v_t = state.get("v", torch.zeros_like(grad))
                m_t= beta_1 * prev_m_t + (1 - beta_1) * grad
                v_t = beta_2 * prev_v_t + (1 - beta_2)*torch.square(grad)
                alpha_t=alpha*(1-beta_2**t)**0.5/(1-beta_1**t)
                p.data -= alpha_t * m_t / (torch.sqrt(v_t) + eps)
                #apply weight decay
                p.data -= alpha * group["weight_decay"] * p.data
                state["t"] = t + 1 
                state["m"] = m_t
                state["v"] = v_t
        return loss