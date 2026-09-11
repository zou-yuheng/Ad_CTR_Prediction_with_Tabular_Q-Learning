import numpy as np
import matplotlib.pyplot as plt

# 1. 加载并检查结构
data = np.load("model_seed42.npz")
Q = data['Q']

print("Q.shape:", Q.shape)
print("Q.ndim :", Q.ndim)
print("Q.size :", Q.size)

# 2. 如果是二维的 (N, num_actions)，取每个 state 的最大 Q 值
if Q.ndim == 2:
    print("→ 二维 Q 表，提取每个 state 的最佳 Q 值...")
    best_Q = Q.max(axis=1)  # 形状: (N,)
elif Q.ndim == 1:
    best_Q = Q
else:
    raise ValueError(f"不支持的维度: {Q.ndim}")

num_bins = len(best_Q)
print(f"→ 共有 {num_bins} 个 state")

# 3. 计算行列
rows = int(np.ceil(np.sqrt(num_bins)))
cols = int(np.ceil(num_bins / rows))
print(f"→ 热力图布局: {rows} 行 x {cols} 列 (需要 {rows*cols} 格)")

# 4. 补齐 NaN
pad = rows * cols - num_bins
if pad > 0:
    best_Q = np.pad(best_Q, (0, pad), constant_values=np.nan)

Q_matrix = best_Q.reshape(rows, cols)

# 5. 绘图
plt.figure(figsize=(14, 10))
im = plt.imshow(Q_matrix, cmap='RdYlGn', aspect='auto')
plt.colorbar(im, label='Q-value')
plt.title(f"Q-Table Heatmap  ({num_bins} states)\n{Q.shape} → best Q per state", fontsize=13)
plt.axis('off')
plt.tight_layout()
plt.savefig("q_heatmap_seed42.png", dpi=150)
print("\n✅ 热力图已保存: q_heatmap_seed42.png")