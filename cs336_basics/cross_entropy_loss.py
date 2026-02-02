from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.bpe import *
from cs336_basics.bpe_tokenizer import *
from cs336_basics.layer import *
from cs336_basics.TransformerBlock import *
from cs336_basics.TransformerLM import *

# calculate cross entropy loss
def cross_entropy_loss(
    logits: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    x_max = torch.max(logits, dim=-1, keepdim=True).values
    x=logits - x_max
    log_x = x - torch.log(torch.sum(torch.exp(x), dim=-1, keepdim=True))
    
    return -torch.mean(torch.gather(log_x, -1, targets.unsqueeze(-1)))
    