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
from cs336_basics.cross_entropy_loss import *
from cs336_basics.optimizer import *
from cs336_basics.clip_gradient import *
from cs336_basics.get_batch import *

# check and save model checkpoint
def save_checkpoint(model, optimizer, iteration, out) -> None:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,

    }
    torch.save(checkpoint, out)

def load_checkpoint(src,model, optimizer) -> tuple[int, Any]:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    
    return iteration