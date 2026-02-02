from cs336_basics.bpe import *
from cs336_basics.bpe_fast_heapq import *
import time
start_time = time.time()
bpe_trainer=BPE_Trainer()
vocab, merges = bpe_trainer.train(
        input_path='lfs-data/owt_train.txt',
            vocab_size=10000,
            special_tokens=["<|endoftext|>","<|endoftext|><|endoftext|>"]
    )
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")
print("Vocabulary Size:", len(vocab))