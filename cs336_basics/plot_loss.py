import os, glob
import torch
import matplotlib.pyplot as plt

def load_run(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    log = ckpt["train_log"]
    meta = log.get("meta", {})
    train = log.get("train", [])
    valid = log.get("valid", [])
    return meta, train, valid

def extract_xy_train(train):
    steps = [r["step"] for r in train if "step" in r and "loss" in r]
    loss  = [r["loss"] for r in train if "step" in r and "loss" in r]
    lr    = [r.get("lr", None) for r in train if "step" in r and "loss" in r]
    tsec  = [r.get("time", None) for r in train if "step" in r and "loss" in r]
    return steps, loss, lr, tsec

def extract_xy_valid(valid):
    steps = [r["step"] for r in valid if "step" in r and "loss" in r]
    loss  = [r["loss"] for r in valid if "step" in r and "loss" in r]
    tsec  = [r.get("time", None) for r in valid if "step" in r and "loss" in r]
    return steps, loss, tsec

def plot_group(ax_train, ax_lr, ax_val, ckpt_paths, title_suffix=""):
    for ckpt in ckpt_paths:
        meta, train, valid = load_run(ckpt)
        label = f'LR={meta.get("lr_initial", "unk")}'
        tr_s, tr_loss, tr_lr, tr_t = extract_xy_train(train)
        va_s, va_loss, va_t = extract_xy_valid(valid)

        ax_train.plot(tr_s, tr_loss, label=label)

        # Learning Rate
        if all(v is not None for v in tr_lr):
            ax_lr.plot(tr_s, tr_lr, label=label)
        else:
            # 如果你没记录 lr，这里可以用 get_lr_cosine 重算（需要你自己 import）
            pass

        # Validation loss vs relative time (min)
        if len(va_t) > 0 and all(v is not None for v in va_t):
            ax_val.plot([t/60 for t in va_t], va_loss, marker="o", linewidth=1, label=label)
        else:
            # 没 time 的话也能画 step vs val_loss
            ax_val.plot(va_s, va_loss, marker="o", linewidth=1, label=label)

    ax_train.set_title("Training Loss" + title_suffix)
    ax_train.set_xlabel("Training Step")
    ax_train.set_ylabel("Loss")
    ax_train.grid(True, alpha=0.3)

    ax_lr.set_title("Learning Rate" + title_suffix)
    ax_lr.set_xlabel("Training Step")
    ax_lr.set_ylabel("LR")
    ax_lr.set_yscale("log")
    ax_lr.grid(True, alpha=0.3)

    ax_val.set_title("Validation Loss" + title_suffix)
    ax_val.set_xlabel("Relative Time (min)")
    ax_val.set_ylabel("Loss")
    ax_val.grid(True, alpha=0.3)

# 例：把两组实验分别放两个目录
group1 = sorted(glob.glob("runs_5k/lr_*/checkpoint.pth"))     # 上排
group2 = sorted(glob.glob("runs_10k/lr_*/checkpoint.pth"))    # 下排

fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=150)

plot_group(axes[0,0], axes[0,1], axes[0,2], group1, title_suffix="")
plot_group(axes[1,0], axes[1,1], axes[1,2], group2, title_suffix="")

# 统一 legend：放在图外或底部
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)

plt.tight_layout(rect=[0,0.08,1,1])
plt.show()

