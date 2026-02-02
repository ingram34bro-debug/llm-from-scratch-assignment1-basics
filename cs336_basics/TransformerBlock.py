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

# transformer block
class TransformerBlock(nn.Module):
    def __init__(self, d_model:int , num_heads:int , d_ff:int,max_seq_len: int,
        rope_theta: float = 10_000.0,
        use_rope: bool = True,
        device=None,
        dtype=None,
    ) -> None: 
        super().__init__()
        kwargs = {'device': device, 'dtype': dtype}
        self.norm1 = RNSNorm(d_model, **kwargs)

        self.attn=Multihead_Self_Attention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            use_rope=use_rope,**kwargs)
        
       
        self.norm2 = RNSNorm(d_model, **kwargs)
        self.ff=SwiGLU(d_model=d_model,d_ff=d_ff,**kwargs)

    def forward(self, x: Tensor,token_positions:torch.Tensor | None = None) -> Tensor:
        out=self.attn(self.norm1(x),token_positions=token_positions)
        x= x+out
        x = x + self.ff(self.norm2(x))
        return x