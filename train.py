# -*- coding: utf-8 -*-
"""训练主流程：加载数据 -> 多 epoch 训练 -> 统计信息 -> 保存模型。"""
import time
import numpy as np
from env import load_and_preprocess
from agent import TabularQLearningAgent


def main():
    np.random.seed(42)
    start_time = time.time()

    # 1. 加载数据
    (train_states, train_clicks, test_states, test_clicks, global_ctr,
     train_banner_pos_ctr, test_banner_pos) = load_and_preprocess('train.csv')
    print(f"数据加载完成，耗时 {time.time()-start_time:.2f}s")
    print(f"全局 CTR: {global_ctr:.4f}")
    print(f"训练集: {len(train_states)} 条, 测试集: {len(test_states)} 条")

    # 2. 训练：多 epoch，每个 epoch 随机打乱
    agent = TabularQLearningAgent(n_buckets=512)
    n_epochs = 5
    for epoch in range(n_epochs):
        perm = np.random.permutation(len(train_states))
        agent.train(train_states[perm], train_clicks[perm])
        print(f"Epoch {epoch+1}/{n_epochs} 完成")

    print(f"训练完成，总耗时 {time.time()-start_time:.2f}s")

    # 3. 统计信息
    active_buckets = int(np.sum(agent.impression_sum > 0))
    low_sample_buckets = int(np.sum((agent.impression_sum > 0) & (agent.impression_sum < 5)))
    print(f"激活桶数: {active_buckets} / 512")
    print(f"低样本桶(<5)数: {low_sample_buckets}")

    # 4. 保存模型
    np.savez('model.npz',
             Q=agent.Q, count=agent.count, click_sum=agent.click_sum,
             impression_sum=agent.impression_sum, global_ctr=global_ctr)
    print(f"模型已保存，总耗时 {time.time()-start_time:.2f}s")


if __name__ == '__main__':
    main()
