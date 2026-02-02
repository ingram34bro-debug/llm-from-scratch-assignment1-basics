from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from jaxtyping import Float, Int

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    
    # 1. 确定随机采样的合法起始位置范围
    max_idx = len(dataset) - context_length 
    
    # 2. 随机生成 batch_size 个起始索引
    ix = torch.randint(0, max_idx , (batch_size,))
    x = torch.stack([torch.from_numpy((dataset[i:i+context_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((dataset[i+1:i+context_length+1]).astype(np.int64)) for i in ix])

        
    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y