from __future__ import annotations

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import json
import tqdm
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
from cs336_basics.check_and_load import *

#增加存储loss的部分
def new_save_checkpoint(model, optimizer, iteration, loss,out) -> None:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        "loss": loss if loss is not None else [],

    }
    torch.save(checkpoint, out)

#增加读取loss的部分
def new_load_checkpoint(src,model, optimizer) ->  tuple[int, Any]:
    checkpoint = torch.load(src)
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    loss=checkpoint.get("loss", [])
    return iteration,loss

def SingleTrain(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
            data: tuple[Tensor, Tensor], T: int, lrarg: tuple[float, float, int, int]) -> float:
    lr = get_lr_cosine(T, *lrarg)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    model.train()
    x, y = data
    logits = model(x)
    logits=rearrange(logits,'b c v -> (b c) v')
    targets = rearrange(y,'b c -> (b c)')
    #loss = cross_entropy_loss(logits,targets)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    
    #gradient_clipping(model.parameters(), max_norm)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()

@torch.no_grad() # 关键：禁用梯度计算，节省显存和计算量
def estimate_loss(model, dataset, batch_size, context_length, device, eval_iters=10):
    model.eval() # 切换到评估模式（关闭 Dropout 等）
    losses = []
    for _ in range(eval_iters):
        # 获取验证集随机 batch
        x, y = get_batch(dataset, batch_size, context_length, device)
        logits = model(x)
        
        # 维度变换以匹配 CrossEntropy
        logits = rearrange(logits, 'b c v -> (b c) v')
        targets = rearrange(y, 'b c -> (b c)')
        
        loss = torch.nn.functional.cross_entropy(logits, targets)
        losses.append(loss.item())
    
    model.train() # 切换回训练模式
    return np.mean(losses)

def _innerTrain(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: npt.NDArray,
    dataset_valid:npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    start_step: int,
    lrarg: tuple[float, float, int, int],
    max_iters: int,
) -> float:
    
    # 进度条，从T进度回复
    progress = tqdm.tqdm(
        range(start_step, max_iters),
        initial=start_step,
        total=max_iters,
        desc="Training",
        unit="iter",
    )
    
    losses = []
    val_losses=[]
    try:
        for T in progress:
            data = get_batch(dataset, batch_size, context_length, device)
            loss = SingleTrain(model, optimizer, data, T, lrarg)
            losses.append(loss)
            if T % 100 == 0: 
                print(np.mean(losses[T//100-1:T//100]))
                val_loss = estimate_loss(model, dataset_valid, batch_size, context_length, device)
                print(f"Step {T}: Train Loss {loss:.4f}, Val Loss {val_loss:.4f}")
                val_losses.append((T, val_loss)) # 记录步数和对应的验证损失
        success = True
    except BaseException as e:
        progress.close()
        print(f"Training interrupted at step {T} due to exception: {e}")
        success = False
    
    return success, T, losses,val_loss

def train(path: str,dataset_path: str,vocab_size:int=10000,device: str="cpu",dtype=None,
          batch_size: int=32,context_length: int=256,
          T: int=1000,lrarg: tuple[float, float, int, int]=None):
    dataset = np.memmap(
        dataset_path,
        dtype=np.uint16,
        mode="r",
    )
    factory_kwargs = {"device": device, "dtype": torch.bfloat16}
    model = TransformerLM(
        vocab_size=vocab_size,context_length=context_length,
        d_model=512,
        num_layers=4,
        num_heads=32,
        d_ff=1344,
        rope_theta=10000,
        **factory_kwargs
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    start_step = 0
    if lrarg is None:
        lrarg =[1e-3,1e-5,1000,10000] #如果没有参数，设置一个lr参数

    if os.path.exists(path):
        start_step, prev_losses = new_load_checkpoint(path, model, optimizer)
        print(f"Resumed from checkpoint at step {start_step}.")
    else:
        prev_losses = []
    
    success, final_step, losses = _innerTrain(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
        start_step=start_step,
        lrarg=lrarg,
        max_iters=T,
    )
    
    new_save_checkpoint(model, optimizer, final_step, loss=prev_losses+losses,out=path,)

if __name__=="__main__":
    
    
    train(
        path='checkpoint',
        dataset_path='./lfs-data/vocab_ts_valid.npy',
        vocab_size=10000,
        dtype=torch.bfloat16,
        batch_size=32,
        context_length=256,
        T=10000,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )