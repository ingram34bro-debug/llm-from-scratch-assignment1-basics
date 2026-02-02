import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(loss_history, save_path="loss_curve.png", window_size=50):
    """
    绘制训练 Loss 曲线
    
    Args:
        loss_history: list[float], 训练过程中的 loss 记录
        save_path: str, 图片保存路径
        window_size: int, 滑动平均的窗口大小（用于平滑曲线，方便观察趋势）
    """
    if not loss_history:
        print("Loss history 为空，无法绘图。")
        return

    plt.figure(figsize=(10, 6))
    
    # 绘制原始 Loss（浅色）
    plt.plot(loss_history, alpha=0.3, color='blue', label='Original Loss')
    
    # 绘制滑动平均曲线（深色，平滑处理）
    if len(loss_history) > window_size:
        # 计算移动平均
        smoothed_loss = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size - 1, len(loss_history)), smoothed_loss, color='red', linewidth=2, label=f'Moving Average (window={window_size})')

    plt.title("Training Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 保存并显示
    plt.savefig(save_path)
    plt.show()
    print(f"Loss curve saved to {save_path}")

# 使用示例：
# _, prev_losses = new_load_checkpoint('checkpoint.pth', model, optimizer)
# plot_loss(prev_losses)