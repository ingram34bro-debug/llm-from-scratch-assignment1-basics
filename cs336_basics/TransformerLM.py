from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
import torch
from torch import Tensor
from torch import nn
from torch.nn import init
from einops import rearrange, einsum
from cs336_basics.layer import *
from cs336_basics.TransformerBlock import *


def copy_param(target: torch.Tensor, source: torch.Tensor) -> None:
    """
    Copy `source` into `target` in-place, transposing `source` if that
    is what makes the shapes line up.
    """
    if source.shape == target.shape:
        target.data.copy_(source)
    elif source.T.shape == target.shape:
        target.data.copy_(source.T)
    else:
        raise ValueError(f"Shape mismatch: cannot load parameter of shape {source.shape} "
                         f"into tensor of shape {target.shape}")
# transformer language model
class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.token_embedding = Embedding(vocab_size, d_model, **factory_kwargs)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff,max_seq_len=context_length, rope_theta=rope_theta, **factory_kwargs)
            for _ in range(num_layers)
        ])
        self.ln_f = RNSNorm(d_model, **factory_kwargs)
        self.output_linear = Linear(d_model, vocab_size, **factory_kwargs)
        self.context_length = context_length
        self.d_model = d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        assert seq_len <= self.context_length, "Sequence length exceeds maximum"

        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(token_ids) 

        for layer in self.layers:
            x = layer(x, token_positions=positions)

        x = self.ln_f(x)
        logits = self.output_linear(x)
        return logits