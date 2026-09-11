# -*- coding: utf-8 -*-
"""多策略多展示率评估 + 可视化。

三种策略：RL Agent / Random Baseline / Rule Baseline
在每个展示率下选取相同数量的样本，确保 CTR 对比公平。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，安全保存图片
import matplotlib.pyplot as plt


def evaluate_all(test_states, test_clicks, agent, global_ctr,
                 train_banner_pos_ctr, test_banner_pos, display_ratios):
    """评估 RL / Random / Rule 三种策略在多个展示率下的 CTR。

    参数:
        test_states: 测试集哈希桶 ID
        test_clicks: 测试集点击标签
        agent: 训练好的 Q-learning 智能体
        global_ctr: 全局 CTR
        train_banner_pos_ctr: dict, 训练集每个 banner_pos 的历史 CTR
        test_banner_pos: np.array, 测试集每条样本的 banner_pos
        display_ratios: list, 评估展示率列表

    返回:
        results: dict, {ratio: {'RL': ctr, 'Random': ctr, 'Rule': ctr}}
    """
    n = len(test_clicks)
    results = {}

    # (a) RL Agent：按桶平滑得分降序
    smooth_scores = agent.get_smooth_scores(global_ctr, lambda_prior=10)
    test_rl_scores = smooth_scores[test_states]
    rl_order = np.argsort(-test_rl_scores, kind='stable')

    # (c) Rule Baseline：测试样本 banner_pos 对应训练集历史 CTR
    rule_scores = np.array([train_banner_pos_ctr.get(bp, global_ctr)
                            for bp in test_banner_pos], dtype=float)
    rule_order = np.argsort(-rule_scores, kind='stable')

    # (b) Random Baseline：固定种子，重复 10 次取平均
    rng = np.random.RandomState(42)

    for ratio in display_ratios:
        k = int(round(n * ratio))
        # RL
        rl_ctr = float(test_clicks[rl_order[:k]].mean()) if k > 0 else 0.0
        # Rule
        rule_ctr = float(test_clicks[rule_order[:k]].mean()) if k > 0 else 0.0
        # Random
        if k > 0:
            ctrs = []
            for _ in range(10):
                idx = rng.choice(n, k, replace=False)
                ctrs.append(test_clicks[idx].mean())
            random_ctr = float(np.mean(ctrs))
        else:
            random_ctr = 0.0

        results[ratio] = {'RL': rl_ctr, 'Random': random_ctr, 'Rule': rule_ctr}

    # 4.5 100% 展示率验证
    if 1.0 in display_ratios:
        r = results[1.0]
        for name, ctr in r.items():
            if abs(ctr - global_ctr) > 0.01:
                print(f"[警告] 100% 展示率下 {name} CTR={ctr:.4f} 与全局 CTR={global_ctr:.4f} 偏差过大，可能存在 bug")
        if not (abs(r['RL'] - r['Random']) < 1e-9 and abs(r['RL'] - r['Rule']) < 1e-9):
            print(f"[警告] 100% 展示率下三策略 CTR 不一致: "
                  f"RL={r['RL']}, Random={r['Random']}, Rule={r['Rule']}")

    # 4.4 控制台打印每个展示率下 RL 相对 Random 的 Lift
    print("[评估结果]")
    for ratio in display_ratios:
        r = results[ratio]
        lift = r['RL'] / r['Random'] if r['Random'] > 0 else float('inf')
        print(f"展示率={int(ratio*100)}%:  "
              f"RL={r['RL']:.4f}, Random={r['Random']:.4f}, Rule={r['Rule']:.4f}, Lift={lift:.2f}")

    return results


def plot_results(results, display_ratios):
    """绘制 CTR vs 展示率折线图并保存为 ctr_vs_ratio.png。"""
    rl = [results[r]['RL'] for r in display_ratios]
    rand = [results[r]['Random'] for r in display_ratios]
    rule = [results[r]['Rule'] for r in display_ratios]
    x = [r * 100 for r in display_ratios]

    plt.figure(figsize=(8, 5))
    plt.plot(x, rl, marker='o', linestyle='-', color='#d62728', label='RL Agent')
    plt.plot(x, rand, marker='s', linestyle='--', color='#1f77b4', label='Random')
    plt.plot(x, rule, marker='^', linestyle='-.', color='#2ca02c', label='Rule')
    plt.xlabel('Display Ratio (%)')
    plt.ylabel('CTR')
    plt.title('CTR vs Display Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ctr_vs_ratio.png', dpi=150)
    print("图表已保存: ctr_vs_ratio.png")
