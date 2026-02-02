from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.hf_tokenizer import *
from cs336_basics.layer import *
from cs336_basics.TransformerBlock import *
from cs336_basics.TransformerLM import *
from cs336_basics.cross_entropy_loss import *
from cs336_basics.optimizer import *
from cs336_basics.clip_gradient import *
from cs336_basics.get_batch import *
from cs336_basics.check_and_load import *
from cs336_basics.bpe_fast_heapq import *
import time


start_time = time.time()
train=BPE_Trainer()
vocab, merges = train.train(
        input_path='lfs-data/TinyStoriesV2-GPT4-train.txt',
        vocab_size=10_000,
        special_tokens=["<|endoftext|>","<|endoftext|><|endoftext|>"],
    )
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")
start_time = time.time()
tokenizer=Tokenizer(vocab=vocab, merges=merges, special_tokens=["<|endoftext|>","<|endoftext|><|endoftext|>"])
end_time = time.time()
print(f"Tokenizer completed in {end_time - start_time:.2f} seconds")

start_time=time.time()
text=read_text('lfs-data/TinyStoriesV2-GPT4-train.txt')
vocab_int=tokenizer.encode(text)
end_time = time.time()
print(f"Tokenizer train text completed in {end_time - start_time:.2f} seconds")
# 创建一个定长的字节数组 (S 代表 bytes，max_len 是长度)
# 注意：必须按 key 的顺序排列
vocab_array_train = np.array(vocab_int, dtype=np.uint16)
np.save('./lfs-data/vocab_ts_train.npy', vocab_array_train)

text=read_text('lfs-data/TinyStoriesV2-GPT4-valid.txt')
vocab_int=tokenizer.encode(text)


# 创建一个定长的字节数组 (S 代表 bytes，max_len 是长度)
# 注意：必须按 key 的顺序排列
vocab_array_valid = np.array(vocab_int, dtype=np.uint16)
np.save('./lfs-data/vocab_ts_valid.npy', vocab_array_valid)
print(f"模型已保存至/lfs-data")