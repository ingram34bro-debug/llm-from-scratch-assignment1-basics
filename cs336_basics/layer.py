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

# linear module
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None) :
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        # 线性化方法
        std = (2.0/(in_features + out_features))**0.5
        init.trunc_normal_(self.weight, mean=0.0, std=std,a=-3.0*std,b=3.0*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum (self.weight,x,'... d_out d_in, ... d_in -> ... d_out')
    

# embedding module
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        std =1
        init.trunc_normal_(self.weight, mean=0.0, std=std,a=-3.0*std,b=3.0*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
# RNSNorm
class RNSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.ones((d_model,), **factory_kwargs))
        
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS = (x.pow(2).mean(dim=-1, keepdim=True)+ self.eps).sqrt() 
        x_normal= x / RMS
        result = x_normal * self.weight
        return result.to(in_dtype)
    
# Swish activation function    
def silu(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return x / (1 + torch.exp(-x))

def glu(a: Tensor, b: Tensor) -> Tensor:
    return a * b

def get_dff(d_model: int) -> int:
        a=(8 * d_model)/3
        dff= int((a+32)//64)*64 #四舍五入
        return dff
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.linear1= Linear(d_model,d_ff,**factory_kwargs) #w1
        self.linear2= Linear(d_ff,d_model,**factory_kwargs) #w2
        self.linear3= Linear(d_model,d_ff,**factory_kwargs) #w3

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        w1x = self.linear1(x)
        w3x = self.linear3(x)
        new=glu(silu(w1x),w3x)
        return self.linear2(new)
    
# RoPE
class RoPE(nn.Module):
    def __init__(self, rope_theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.rope_theta = rope_theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        freq =1.0/ (self.rope_theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        pos_seq = torch.arange(0, max_seq_len, device=device).float()
        freqs=torch.outer(pos_seq, freq)
        self.register_buffer('cos_cached', torch.cos(freqs), persistent=False)
        self.register_buffer('sin_cached', torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_pos =self.cos_cached[token_positions]
        sin_pos =self.sin_cached[token_positions]
        x1, x2 = x[..., ::2], x[..., 1::2] # x1: even, x2: odd

        x_rotated = x1 * cos_pos - x2 * sin_pos
        y_rotated = x1 * sin_pos + x2 * cos_pos
        
        out = torch.empty_like(x)
        out[..., ::2] = x_rotated
        out[..., 1::2] = y_rotated
        return out

def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    x_max = torch.max(in_features, dim=dim, keepdim=True).values
    x_eps= torch.exp(in_features - x_max)
    return x_eps / torch.sum(x_eps, dim=dim, keepdim=True)

# : Implement scaled dot-product attention
class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self, d_k: int, device=None, dtype=None):
        super().__init__()
        self.d_k = d_k
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:
        scores = einsum(Q,K,'... queries d_k,... keys d_k -> ... queries keys') / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        softmax_scores = softmax(scores, dim=-1)
        output = einsum(softmax_scores,V,'... queries keys,... keys d_v -> ... queries d_v')
        return output
    
# multi-head self-attention
class Multihead_Self_Attention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,max_seq_len: int,
                rope_theta: float =10000.0,use_rope: bool = True,device=None,dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.use_rope = use_rope
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        # Initialize projection weights
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.q_proj,self.k_proj,self.v_proj,self.o_proj = [Linear(d_model, d_model, **factory_kwargs) for _ in range(4)]
        self.atten= Scaled_Dot_Product_Attention(self.d_k, device=device, dtype=dtype)

        mask = torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool, device=device))
        self.register_buffer('mask', mask, persistent=False)
        if use_rope:
            self.rope = RoPE(
                rope_theta=self.rope_theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

    def forward(self, 
                x: Float[Tensor, "... sequence_length d_model"], 
                token_positions: Int[Tensor, "... sequence_length"] | None=None
                ) -> Float[Tensor, "... sequence_length d_out"]:
        
        q_out=self.q_proj(x)
        k_out=self.k_proj(x)
        v_out=self.v_proj(x)
        S = x.shape[-2]
        q=rearrange(q_out,'... seq_len (h d_k) -> ... h seq_len d_k',h=self.num_heads)
        k=rearrange(k_out,'... seq_len (h d_k) -> ... h seq_len d_k',h=self.num_heads)
        v=rearrange(v_out,'... seq_len (h d_k) -> ... h seq_len d_k',h=self.num_heads)
        if self.use_rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        output=self.atten(
            Q=q,
            K=k,
            V=v,
            mask=self.mask[:S, :S]
        )
        output=rearrange(output,'... h seq_len d_k -> ... seq_len (h d_k)')
        return self.o_proj(output)