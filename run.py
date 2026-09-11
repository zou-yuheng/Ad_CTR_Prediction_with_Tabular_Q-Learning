# -*- coding: utf-8 -*-
"""一键运行入口。

模式 1: python run.py                        # 单种子默认流程 (seed=42)
模式 2: python run.py --multi-seed           # 多种子鲁棒性验证 (seeds=[42,123,456,789,999])
"""
import argparse
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from env import load_and_preprocess
from agent import TabularQLearningAgent
from evaluate import evaluate_all, plot_results

DISPLAY_RATIOS = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
MULTI_SEEDS = [42, 123, 456, 789, 999]
N_EPOCHS = 5


def run_single(seed, display_ratios, verbose=True):
    """单次完整流程：加载 -> 训练 -> 评估。

    返回:
        per_ratio: dict, {ratio: {'RL': ctr, 'Random': ctr, 'Rule': ctr, 'Lift': lift}}
        info: dict, 其他元信息 (耗时、激活桶数等)
    """
    start = time.time()
    np.random.seed(seed)

    # 1. 数据加载
    (train_states, train_clicks, test_states, test_clicks, global_ctr,
     train_banner_pos_ctr, test_banner_pos) = load_and_preprocess('train.csv', seed=seed)
    if verbose:
        print(f"  [seed={seed}] 数据加载 {time.time()-start:.2f}s | "
              f"全局CTR={global_ctr:.4f} | 训练集={len(train_states)} | 测试集={len(test_states)}")

    # 2. 训练
    agent = TabularQLearningAgent(n_buckets=512)
    for _ in range(N_EPOCHS):
        perm = np.random.permutation(len(train_states))
        agent.train(train_states[perm], train_clicks[perm])
    active_buckets = int(np.sum(agent.impression_sum > 0))
    low_sample_buckets = int(np.sum((agent.impression_sum > 0) & (agent.impression_sum < 5)))
    if verbose:
        print(f"  [seed={seed}] 训练完成 {time.time()-start:.2f}s | "
              f"激活桶={active_buckets}/512 | 低样本(<5)={low_sample_buckets}")

    # 2.1 保存每个种子的模型 (model_seed42.npz, model_seed123.npz, ...)
    np.savez(f'model_seed{seed}.npz',
             Q=agent.Q, count=agent.count, click_sum=agent.click_sum,
             impression_sum=agent.impression_sum, global_ctr=global_ctr)
    if verbose:
        print(f"  [seed={seed}] 模型已保存: model_seed{seed}.npz")

    # 3. 评估
    results = evaluate_all(test_states, test_clicks, agent, global_ctr,
                           train_banner_pos_ctr, test_banner_pos, display_ratios)

    # 为每个展示率补充 Lift
    per_ratio = {}
    for r, v in results.items():
        lift = v['RL'] / v['Random'] if v['Random'] > 0 else float('inf')
        per_ratio[r] = {**v, 'Lift': lift}

    if verbose:
        print(f"  [seed={seed}] 评估完成 总耗时 {time.time()-start:.2f}s")

    info = {
        'seed': seed,
        'elapsed': time.time() - start,
        'active_buckets': active_buckets,
        'low_sample_buckets': low_sample_buckets,
        'global_ctr': global_ctr,
    }
    return per_ratio, info


def print_summary_table(seeds, display_ratios, per_seed):
    """打印每个 seed × 展示率的 RL CTR / Lift 汇总表 + 5 次均值与标准差。"""
    header = ["seed/ratio"]
    for r in display_ratios:
        header.append(f"RL@{int(r*100)}%")
        header.append(f"Lift@{int(r*100)}%")
    header.append("耗时(s)")

    rows = []
    for seed in seeds:
        row = [str(seed)]
        for r in display_ratios:
            v = per_seed[seed][r]
            row.append(f"{v['RL']:.4f}")
            row.append(f"{v['Lift']:.2f}")
        row.append(f"{v.get('_elapsed', 0):.2f}")  # 占位，下面在汇总前会填入
        rows.append(row)

    # 先做数值收集用于均值/标准差
    rl_matrix = np.array([[per_seed[s][r]['RL'] for r in display_ratios] for s in seeds])
    lift_matrix = np.array([[per_seed[s][r]['Lift'] for r in display_ratios] for s in seeds])

    rl_mean = rl_matrix.mean(axis=0)
    rl_std = rl_matrix.std(axis=0, ddof=0)
    lift_mean = lift_matrix.mean(axis=0)
    lift_std = lift_matrix.std(axis=0, ddof=0)

    # 打印 RL CTR 表
    print("\n" + "=" * 90)
    print("汇总表 1: RL CTR  (每行一个seed，最后两行均值/标准差)")
    print("=" * 90)
    rl_hdr = f"{'Seed':<8}" + "".join(f"{'@' + str(int(r*100)) + '%':>10}" for r in display_ratios)
    print(rl_hdr)
    print("-" * 90)
    for i, seed in enumerate(seeds):
        line = f"{seed:<8}" + "".join(f"{rl_matrix[i, j]:>10.4f}" for j in range(len(display_ratios)))
        print(line)
    print("-" * 90)
    line_mean = f"{'Mean':<8}" + "".join(f"{rl_mean[j]:>10.4f}" for j in range(len(display_ratios)))
    line_std = f"{'Std':<8}" + "".join(f"{rl_std[j]:>10.4f}" for j in range(len(display_ratios)))
    print(line_mean)
    print(line_std)

    # 打印 Lift 表
    print("\n" + "=" * 90)
    print("汇总表 2: Lift (RL/Random)  (每行一个seed，最后两行均值/标准差)")
    print("=" * 90)
    lift_hdr = f"{'Seed':<8}" + "".join(f"{'@' + str(int(r*100)) + '%':>10}" for r in display_ratios)
    print(lift_hdr)
    print("-" * 90)
    for i, seed in enumerate(seeds):
        line = f"{seed:<8}" + "".join(f"{lift_matrix[i, j]:>10.2f}" for j in range(len(display_ratios)))
        print(line)
    print("-" * 90)
    line_mean = f"{'Mean':<8}" + "".join(f"{lift_mean[j]:>10.2f}" for j in range(len(display_ratios)))
    line_std = f"{'Std':<8}" + "".join(f"{lift_std[j]:>10.4f}" for j in range(len(display_ratios)))
    print(line_mean)
    print(line_std)
    print("=" * 90)

    return rl_matrix, lift_matrix, rl_mean, rl_std, lift_mean, lift_std


def plot_multi_seed_lift(seeds, display_ratios, lift_matrix, lift_mean,
                         out_path='multi_seed_lift.png'):
    """多个 seed 的 Lift 曲线 + 均值曲线。"""
    x = [r * 100 for r in display_ratios]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(seeds), 10)))

    plt.figure(figsize=(9, 6))
    for i, seed in enumerate(seeds):
        plt.plot(x, lift_matrix[i], marker='o', alpha=0.6,
                 color=colors[i], linestyle='--', linewidth=1.2,
                 label=f'seed={seed}')
    plt.plot(x, lift_mean, marker='X', color='black', linestyle='-',
             linewidth=2.2, markersize=9, label='Mean (5 seeds)')

    plt.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.8)
    plt.xlabel('Display Ratio (%)')
    plt.ylabel('Lift = CTR_RL / CTR_Random')
    plt.title('Multi-Seed Robustness: Lift vs Display Ratio')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"多种子 Lift 图已保存: {out_path}")


def plot_mean_ctr_vs_ratio(display_ratios, rl_mean, random_mean, rule_mean,
                           out_path='ctr_vs_ratio.png'):
    """绘制 5 种子平均的三策略 CTR 曲线 (RL / Random / Rule)。

    与单种子模式共用同一文件名 ctr_vs_ratio.png，方便对比。
    """
    x = [r * 100 for r in display_ratios]

    plt.figure(figsize=(8, 5))
    plt.plot(x, rl_mean, marker='o', linestyle='-', color='#d62728',
             label='RL Agent (mean of 5 seeds)')
    plt.plot(x, random_mean, marker='s', linestyle='--', color='#1f77b4',
             label='Random (mean of 5 seeds)')
    plt.plot(x, rule_mean, marker='^', linestyle='-.', color='#2ca02c',
             label='Rule (mean of 5 seeds)')

    plt.xlabel('Display Ratio (%)')
    plt.ylabel('CTR')
    plt.title('CTR vs Display Ratio (Mean of 5 Seeds)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"三策略均值曲线已保存: {out_path}")


def multi_seed_main(seeds=None):
    """多种子鲁棒性验证：依次 seed=[42,123,456,789,999]，打印汇总，保存汇总图。"""
    if seeds is None:
        seeds = list(MULTI_SEEDS)
    display_ratios = list(DISPLAY_RATIOS)

    total_start = time.time()
    per_seed = {}
    infos = []
    for i, seed in enumerate(seeds, 1):
        print(f"\n[{i}/{len(seeds)}] ====== seed={seed} 开始 ======")
        t0 = time.time()
        per_ratio, info = run_single(seed, display_ratios, verbose=True)
        info['elapsed'] = time.time() - t0
        for r in display_ratios:
            per_ratio[r]['_elapsed'] = info['elapsed']
        per_seed[seed] = per_ratio
        infos.append(info)

    print(f"\n================ 5 次全部完成，总耗时 {time.time()-total_start:.2f}s ================")
    for info in infos:
        print(f"  seed={info['seed']:<4}  激活桶={info['active_buckets']}/512  "
              f"耗时={info['elapsed']:.2f}s  全局CTR={info['global_ctr']:.4f}")

    rl_matrix, lift_matrix, rl_mean, rl_std, lift_mean, lift_std = \
        print_summary_table(seeds, display_ratios, per_seed)

    # 收集 Random / Rule 矩阵并取均值，用于画三策略均值曲线
    random_matrix = np.array([[per_seed[s][r]['Random'] for r in display_ratios] for s in seeds])
    rule_matrix = np.array([[per_seed[s][r]['Rule'] for r in display_ratios] for s in seeds])
    random_mean = random_matrix.mean(axis=0)
    rule_mean = rule_matrix.mean(axis=0)

    plot_multi_seed_lift(seeds, display_ratios, lift_matrix, lift_mean)
    plot_mean_ctr_vs_ratio(display_ratios, rl_mean, random_mean, rule_mean)

    # 可选：把详细结果保存为 CSV，便于后续分析
    try:
        rows = []
        for seed in seeds:
            for r in display_ratios:
                v = per_seed[seed][r]
                rows.append({
                    'seed': seed,
                    'ratio': r,
                    'RL': v['RL'], 'Random': v['Random'], 'Rule': v['Rule'],
                    'Lift': v['Lift']
                })
        pd.DataFrame(rows).to_csv('multi_seed_results.csv', index=False, float_format='%.6f')
        print("多种子明细已保存: multi_seed_results.csv")
    except Exception as e:
        print(f"[提示] 保存 CSV 失败（不影响主流程）: {e}")


def main():
    parser = argparse.ArgumentParser(description="Offline Contextual Bandit 一键入口")
    parser.add_argument('--multi-seed', action='store_true',
                        help='开启多种子鲁棒性验证 (seeds=42,123,456,789,999)')
    args = parser.parse_args()

    if args.multi_seed:
        multi_seed_main()
    else:
        # 原有单种子流程
        total_start = time.time()
        np.random.seed(42)

        (train_states, train_clicks, test_states, test_clicks, global_ctr,
         train_banner_pos_ctr, test_banner_pos) = load_and_preprocess('train.csv')
        print(f"[数据加载] 耗时 {time.time()-total_start:.2f}s")
        print(f"全局 CTR: {global_ctr:.4f}")
        print(f"训练集: {len(train_states)} 条, 测试集: {len(test_states)} 条")

        agent = TabularQLearningAgent(n_buckets=512)
        for epoch in range(N_EPOCHS):
            perm = np.random.permutation(len(train_states))
            agent.train(train_states[perm], train_clicks[perm])
        print(f"[训练完成] {N_EPOCHS} epochs, 耗时 {time.time()-total_start:.2f}s")

        active_buckets = int(np.sum(agent.impression_sum > 0))
        low_sample_buckets = int(np.sum((agent.impression_sum > 0) & (agent.impression_sum < 5)))
        print(f"激活桶数: {active_buckets} / 512")
        print(f"低样本桶(<5)数: {low_sample_buckets}")

        display_ratios = DISPLAY_RATIOS
        results = evaluate_all(test_states, test_clicks, agent, global_ctr,
                               train_banner_pos_ctr, test_banner_pos, display_ratios)
        print(f"[评估完成] 耗时 {time.time()-total_start:.2f}s")

        plot_results(results, display_ratios)
        print(f"[全部完成] 总耗时 {time.time()-total_start:.2f}s")


if __name__ == '__main__':
    main()
