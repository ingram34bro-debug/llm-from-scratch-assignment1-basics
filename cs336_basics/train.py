import torch
from torch import Tensor
import numpy as np

from typing import Optional
from jaxtyping import Bool, Float, Int
import os
import typing
import numpy.typing as npt

from collections.abc import Callable, Iterable
import math

import tqdm
import time
import pickle


from einops import rearrange, einsum

def create_model(meta: dict, device: str) -> torch.nn.Module:
    model = TransformerLM(
        vocab_size = meta["vocab_size"],
        d_model = meta["d_model"],
        num_layers = meta["num_layers"],
        num_heads = meta["num_heads"],
        rope_theta= meta["theta"],
        context_length= meta["max_seq_len"],
        d_ff = meta["d_ff"],
        device=device,
    ).to(device)
    return model

def load_meta(meta_path: str) -> dict:
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta

def MyGetBatch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    data_len = dataset.shape[0]
    ix = np.random.randint(0, data_len - context_length, size=batch_size)

    x = np.empty((batch_size, context_length), dtype=np.int64)
    y = np.empty_like(x)

    for k, i in enumerate(ix):
        x[k] = dataset[i : i + context_length]
        y[k] = dataset[i + 1 : i + context_length + 1]

    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
    )

def MySaveCheckpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,  # ✅ iteration 现在定义为“下一次要开始的 step”
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    train_log: dict,
) -> None:
    checkpoint = {
        "checkpoint_version": 2,
        "iteration": iteration,
        "iteration_is_next_step": True,  # ✅ 新语义标记
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_log": train_log,
    }
    torch.save(checkpoint, out)


def MyLoadCheckpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[str] = None,
):
    checkpoint = torch.load(src, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    raw_iter = int(checkpoint.get("iteration", 0))
    train_log = checkpoint.get("train_log", None)

    # ✅ 兼容旧 checkpoint（旧版本 iteration 是“最后一次尝试的 step / 或最后完成的 step”，语义不稳定）
    if checkpoint.get("iteration_is_next_step", False):
        start_step = raw_iter
    else:
        # 旧版本：用 train_log 推断最后完成的 step
        if isinstance(train_log, dict) and train_log.get("train"):
            # old code: train_log["train"] 每步都 append {"step": T,...}
            last_completed = int(train_log["train"][-1]["step"])
            start_step = last_completed + 1
        else:
            # 没有任何 train 记录时，保守从 raw_iter 开始
            start_step = raw_iter

    return start_step, train_log



# 定义单次训练：

from optimizer import *
from get_batch import *
from TransformerLM import TransformerLM

def SingleTrain(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: tuple[Tensor, Tensor],
    step: int,
    lrarg: tuple[float, float, int, int],
) -> dict:
    lr = get_lr_cosine(step, *lrarg)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    model.train()
    x_batch, y_batch = data

    logits = model(x_batch)                       # (B, S, V)
    logits = rearrange(logits, "b c v -> (b c) v") # (B*S, V)
    targets = rearrange(y_batch, "b c -> (b c)")   # (B*S,)

    loss = torch.nn.functional.cross_entropy(logits, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "lr": float(lr),
        "grad_norm": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
    }


def SingleValidate(model: torch.nn.Module, dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> float:
    model.eval()
    with torch.no_grad():
        data = MyGetBatch(dataset, batch_size, context_length, device)
        x_batch, y_batch = data
        logits = model(x_batch)
        logits = rearrange(logits, "b c v -> (b c) v")
        targets = rearrange(y_batch, "b c -> (b c)")
        loss = torch.nn.functional.cross_entropy(logits, targets)
    return loss.item()
def ValidateAvg(
    model: torch.nn.Module,
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    k: int = 20,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(k):
            x_batch, y_batch = MyGetBatch(dataset, batch_size, context_length, device)
            logits = model(x_batch)                       # (B, S, V)
            logits = rearrange(logits, "b c v -> (b c) v") # (B*S, V)
            targets = rearrange(y_batch, "b c -> (b c)")   # (B*S,)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            losses.append(loss.item())
    return float(sum(losses) / len(losses))

def _innerTrain(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    start_step: int,
    lrarg: tuple[float, float, int, int],
    max_iters: int,
    ckpt_path: str,
    val_dataset: Optional[npt.NDArray] = None,
    train_log: Optional[dict] = None,
    # ✅ 你要求的频率
    log_every: int = 10,
    val_every: int = 200,
    save_every: int = 500,
    val_batches: int = 20,
    # ✅ W&B
    use_wandb: bool = False,
):
    if train_log is None:
        train_log = {"train": [], "valid": []}

    # tqdm 从 start_step 开始
    progress = tqdm.tqdm(
        range(start_step, max_iters),
        initial=start_step,
        total=max_iters,
        desc="Training",
        unit="step",
    )

    start_time = time.time()

    # 让这个变量始终表示“下一步要跑的 step”（断点对齐核心）
    next_step_to_run = start_step

    ex = None
    success = False

    try:
        for step in progress:
            # step == 当前要执行的 step（0-based）
            t0 = time.time()
            data = MyGetBatch(dataset, batch_size, context_length, device)

            # 单步训练（如果这里异常，next_step_to_run 保持为 step，不会错误 +1）
            m = SingleTrain(model, optimizer, data, step, lrarg)

            step_time = time.time() - t0
            elapsed = time.time() - start_time

            # 这一行只在“该 step 成功完成一次 optimizer.step()”后执行
            next_step_to_run = step + 1

            tokens_per_step = batch_size * context_length
            tok_per_sec = tokens_per_step / max(step_time, 1e-9)

            # ========= train log (every 10 steps) =========
            if next_step_to_run % log_every == 0:
                row = {
                    "step": next_step_to_run,
                    "loss": m["loss"],
                    "lr": m["lr"],
                    "grad_norm": m["grad_norm"],
                    "time": elapsed,
                }
                train_log.setdefault("train", []).append(row)

                progress.set_postfix({"loss": m["loss"], "lr": m["lr"]})

                if use_wandb:
                    import wandb
                    wandb.log(
                        {
                            "train/loss": m["loss"],
                            "train/grad_norm": m["grad_norm"],
                            "lr": m["lr"],
                            "perf/step_time_sec": step_time,
                            "perf/tokens_per_sec": tok_per_sec,
                            "time/elapsed_sec": elapsed,
                        },
                        step=next_step_to_run,
                    )

            # ========= valid log (every 200 steps, averaged) =========
            if val_dataset is not None and next_step_to_run % val_every == 0:
                val_loss = ValidateAvg(
                    model=model,
                    dataset=val_dataset,
                    batch_size=batch_size,
                    context_length=context_length,
                    device=device,
                    k=val_batches,
                )

                vrow = {"step": next_step_to_run, "loss": val_loss, "time": elapsed}
                train_log.setdefault("valid", []).append(vrow)

                progress.set_postfix({"loss": m["loss"], "val_loss": val_loss, "lr": m["lr"]})

                if use_wandb:
                    import wandb
                    wandb.log(
                        {
                            "valid/loss": val_loss,
                            "valid/k_batches": val_batches,
                            "time/elapsed_sec": elapsed,
                        },
                        step=next_step_to_run,
                    )

            # ========= checkpoint (every 500 steps) =========
            if next_step_to_run % save_every == 0:
                MySaveCheckpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=next_step_to_run,  # ✅ 保存“下一步要跑的 step”
                    out=ckpt_path,
                    train_log=train_log,
                )
                # 也可以在控制台提示一下：
                progress.write(f"[ckpt] saved at step={next_step_to_run} -> {ckpt_path}")

        success = True

    except BaseException as e:
        ex = e
        # next_step_to_run 在异常时仍然是“下一步要跑的 step”（通常等于当前 step）
        print(f"Training interrupted near step {next_step_to_run} due to exception: {e}")
        success = False

    finally:
        progress.close()

    return success, next_step_to_run, train_log, ex

def Train(
    path: str,
    dataset_path: str,
    meta_path: str,
    val_dataset_path: Optional[str] = None,
    **meta_kwargs,
):
    dataset = np.memmap(dataset_path, dtype=np.uint16, mode="r")
    val_dataset = np.memmap(val_dataset_path, dtype=np.uint16, mode="r") if val_dataset_path else None

    # 读 meta
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
    else:
        meta = {}
        meta.update(meta_kwargs)
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

    # 允许运行时覆盖 meta（你现在只覆盖 device，我建议把下面这些也覆盖）
    for k in [
        "device",
        "use_wandb",
        "wandb_project",
        "wandb_run_name",
        "val_batches",
        "log_every",
        "val_every",
        "save_every",
    ]:
        if k in meta_kwargs:
            meta[k] = meta_kwargs[k]

    device = meta.get("device", "cpu")
    print(meta)

    model = create_model(meta, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # ====== 载入 checkpoint ======
    start_step = 0
    train_log = {"meta": meta, "train": [], "valid": []}

    if os.path.exists(path):
        start_step, loaded_log = MyLoadCheckpoint(path, model, optimizer, device)
        if isinstance(loaded_log, dict):
            train_log = loaded_log
            train_log.setdefault("meta", meta)
            train_log.setdefault("train", [])
            train_log.setdefault("valid", [])
        else:
            train_log = {"meta": meta, "train": [], "valid": []}
        print(f"Resumed from checkpoint. start_step={start_step}")
    else:
        print("No checkpoint found. Start from scratch.")

    # ====== W&B init / resume ======
    use_wandb = bool(meta.get("use_wandb", True))  # 你要求用 W&B，这里默认 True 也行
    if use_wandb:
        import wandb

        project = meta.get("wandb_project", "llm-training")
        name = meta.get("wandb_run_name", None)

        # 断点续训对齐：同一个 run id 续跑
        run_id = train_log.get("wandb_run_id", None)

        wandb.init(
            project=project,
            name=name,
            id=run_id,              # None -> new run；有值 -> resume
            resume="allow",         # 允许续跑
            config=meta,
        )

        # 新建 run 时写回 id（下一次 checkpoint 就会保存它）
        train_log["wandb_run_id"] = wandb.run.id

        # 可选：显示当前 resume 信息
        print(f"W&B: project={project}, run_id={wandb.run.id}, resume_from_step={start_step}")

    # ====== 训练（10/200/500 + 平均验证）======
    log_every = int(meta.get("log_every", 10))
    val_every = int(meta.get("val_every", 200))
    save_every = int(meta.get("save_every", 500))
    val_batches = int(meta.get("val_batches", 20))

    success, next_step, train_log, e = _innerTrain(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        batch_size=meta["batch_size"],
        context_length=meta["max_seq_len"],
        device=device,
        start_step=start_step,
        lrarg=(
            meta["lr_initial"],
            meta["lr_final"],
            meta["lr_warmup_iters"],
            meta["max_iters"],
        ),
        max_iters=meta["max_iters"],
        ckpt_path=path,
        val_dataset=val_dataset,
        train_log=train_log,
        log_every=log_every,
        val_every=val_every,
        save_every=save_every,
        val_batches=val_batches,
        use_wandb=use_wandb,
    )

    # ====== 最终保存（确保最后状态落盘）======
    MySaveCheckpoint(
        model=model,
        optimizer=optimizer,
        iteration=next_step,
        out=path,
        train_log=train_log,
    )

    if use_wandb:
        import wandb
        wandb.log({"train/finished": 1, "final/next_step": next_step}, step=next_step)
        wandb.finish()

    return not isinstance(e, KeyboardInterrupt)


if __name__ == "__main__":
    
    import cProfile
    import pstats
    
    pr = cProfile.Profile()
    pr.enable()

    if_shutdown = Train(
    path="/home/std7/extend/llm-from-scratch-assignment1-basics/my_module/lr_3e-2/checkpoint_TinyStores.pth",
    dataset_path="/home/std7/extend/lfs-data/TinyStoriesV2-GPT4-train.npy",
    meta_path="/home/std7/extend/llm-from-scratch-assignment1-basics/my_module/lr_3e-2/meta_TinyStores.pkl",
    val_dataset_path="/home/std7/extend/lfs-data/TinyStoriesV2-GPT4-valid.npy",  # 例子：你应换成真正 valid
    vocab_size=10000,
    num_layers=4,
    d_model=512,
    num_heads=16,
    d_ff=1344,
    theta=10000,
    max_seq_len=256,
    batch_size=32,
    lr_initial=3e-2,
    lr_final=1e-5,
    lr_warmup_iters=1000,
    max_iters=40000,
    device="cuda" if torch.cuda.is_available() else "cpu",

    # ✅ W&B + 频率（也可不传，默认就是 10/200/500/20）
    use_wandb=True,
    wandb_project="llm-from-scratch",
    wandb_run_name="baseline",
    log_every=10,
    val_every=200,
    save_every=500,
    val_batches=20,
)
    pr.disable()

    stats = pstats.Stats(pr)
    stats.strip_dirs()
    stats.sort_stats("cumtime")
    stats.print_stats(30)