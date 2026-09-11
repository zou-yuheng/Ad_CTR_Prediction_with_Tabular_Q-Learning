# -*- coding: utf-8 -*-
"""Tabular Q-learning 智能体：增量式点击率均值估计器。"""
import numpy as np


class TabularQLearningAgent:
    def __init__(self, n_buckets=512):
        self.n_buckets = n_buckets
        self.Q = np.zeros((n_buckets, 2))            # Q[s][0]=0, Q[s][1]=CTR 估计
        self.count = np.zeros(n_buckets)             # 每个桶访问计数
        self.click_sum = np.zeros(n_buckets)         # 每个桶点击总数
        self.impression_sum = np.zeros(n_buckets)    # 每个桶展示总数

    def train(self, states, clicks):
        """对每条样本无条件更新 Q(s,1)。

        Offline Bandit 中不使用 epsilon-greedy：探索动作 0 无法获取真实奖励，
        会浪费训练样本，因此直接对所有展示样本更新 Q(s,1)。
        """
        Q = self.Q
        count = self.count
        click_sum = self.click_sum
        impression_sum = self.impression_sum
        for s, click in zip(states, clicks):
            s = int(s)
            count[s] += 1
            click_sum[s] += click
            impression_sum[s] += 1
            # 递减学习率，等价于精确均值估计
            alpha = 1.0 / count[s]
            Q[s, 1] += alpha * (click - Q[s, 1])
            # Q[s][0] 始终为 0，不更新

    def get_smooth_scores(self, global_ctr, lambda_prior=10):
        """拉普拉斯平滑评分，先验均值必须与全局 CTR 对齐。

        score = (click_sum + lambda_prior * global_ctr) / (impression_sum + lambda_prior)
        对于从未被访问的桶（impression_sum=0），平滑得分 = global_ctr。
        """
        scores = (self.click_sum + lambda_prior * global_ctr) / (self.impression_sum + lambda_prior)
        return scores
