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
from cs336_basics.tokenizer_fast import *
import time
import pickle

start_time = time.time()
train=BPE_Trainer()
vocab, merges = train.train(
        input_path='/home/std7/extend/lfs-data/TinyStoriesV2-GPT4-train.txt',
        vocab_size=10_000,
        special_tokens=["<|endoftext|>","<|endoftext|><|endoftext|>"],
    )
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")


# if __name__ == "__main__":
#     start_time = time.time()
#     TokenizeData(
#         "/home/std7/extend/lfs-data/TinyStoriesV2-GPT4-valid.txt",
#         "/home/std7/extend/lfs-data",
#         vocab_path='/home/std7/extend/lfs-data/ts_bpe_vocab.pkl',
#         merges_path='/home/std7/extend/lfs-data/ts_bpe_merges.pkl',
#         special_tokens=[
#             b"<|endoftext|>",
#         ],
#     )
#     end_time = time.time()
#     print(f"Tokenizer completed in {end_time - start_time:.2f} seconds")





