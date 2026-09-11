import numpy as np
import matplotlib.pyplot as plt

def plot_q_and_count(npz_path, save_path):
    # 1. 加载模型
    data = np.load(npz_path)
    Q = data['Q']          # 形状: (512, 2)
    counts = data['count'] # 形状: (512,)

    # 2. 提取最佳 Q 值（原始 512 个）
    if Q.ndim == 2:
        best_Q = Q.max(axis=1)
    else:
        best_Q = Q

    num_bins = len(best_Q)  # 512

    # 3. 计算布局
    rows = int(np.ceil(np.sqrt(num_bins)))  # 23
    cols = int(np.ceil(num_bins / rows))    # 23
    pad = rows * cols - num_bins             # 17

    # 4. 先统计（用原始 512 个数据）
    valid_Q = best_Q[~np.isnan(best_Q)]
    threshold_Q = np.percentile(valid_Q, 80) if len(valid_Q) else 0.5
    threshold_count = 5
    
    suspicious = np.sum((best_Q > threshold_Q) & (counts < threshold_count))
    print(f"  Q 阈值(80%分位): {threshold_Q:.4f}")
    print(f"  Count 阈值: {threshold_count}")
    print(f"  ⚠️ 高Q值但低访问(state未充分探索): {suspicious} 个")

    # 5. 再 pad（两个数组同时补 NaN）
    if pad > 0:
        best_Q_padded = np.pad(best_Q, (0, pad), constant_values=np.nan)
        counts_padded = np.pad(counts, (0, pad), constant_values=np.nan)
    else:
        best_Q_padded = best_Q
        counts_padded = counts

    # count 取 log，NaN 保持 NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        log_counts = np.where(np.isnan(counts_padded), np.nan, np.log1p(counts_padded))

    Q_matrix = best_Q_padded.reshape(rows, cols)
    Count_matrix = log_counts.reshape(rows, cols)

    # 6. 绘图
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # 左图：Q 值热力图
    im1 = axes[0].imshow(Q_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    fig.colorbar(im1, ax=axes[0], label='Q-value')
    axes[0].set_title(f"Q-Table Heatmap ({num_bins} states)\n(512, 2) → best Q per state", fontsize=12)
    axes[0].axis('off')

    # 右图：Count 热力图
    im2 = axes[1].imshow(Count_matrix, cmap='hot', aspect='auto')
    fig.colorbar(im2, ax=axes[1], label='log(1 + count)')
    axes[1].set_title(f"State Visit Count ({num_bins} states)\nlog scale", fontsize=12)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 对比图已保存: {save_path}")


if __name__ == "__main__":
    print("==================================================")
    print("📊 Q-Table vs Count 对比分析")
    print("==================================================")
    plot_q_and_count("model_seed42.npz", "q_vs_count_comparison.png")