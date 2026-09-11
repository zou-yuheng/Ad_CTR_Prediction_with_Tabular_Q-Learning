# -*- coding: utf-8 -*-
"""数据加载与特征工程：随机切分、长尾截断、确定性哈希映射、banner_pos 历史 CTR。"""
import hashlib
import numpy as np
import pandas as pd

COLUMNS = ['id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
           'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
           'device_ip', 'device_model', 'device_type', 'device_conn_type',
           'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']

N_BUCKETS = 512


def _hash_state(hour_bucket, banner_pos, site_id_group, app_category_group):
    """确定性哈希：md5 取模，禁止使用内置 hash()。"""
    state_str = f"{hour_bucket}|{banner_pos}|{site_id_group}|{app_category_group}"
    return int(hashlib.md5(state_str.encode('utf-8')).hexdigest(), 16) % N_BUCKETS


def load_and_preprocess(file_path, random_state=42, seed=None):
    """读取 -> 随机切分 -> 特征工程 -> 哈希映射。

    参数:
        file_path: 数据文件路径
        random_state: 随机切分种子（兼容旧接口）
        seed: 当不为 None 时覆盖 random_state，便于多次调用传入不同种子

    返回:
        train_states: np.array, 训练集哈希桶 ID (shape: [80000])
        train_clicks: np.array, 训练集点击标签 (shape: [80000])
        test_states: np.array, 测试集哈希桶 ID (shape: [20000])
        test_clicks: np.array, 测试集点击标签 (shape: [20000])
        global_ctr: float, 全局平均点击率
        train_banner_pos_ctr: dict, 训练集每个 banner_pos 的历史 CTR
        test_banner_pos: np.array, 测试集每条样本的 banner_pos
    """
    if seed is not None:
        random_state = seed

    # 1.1 数据读取：所有列统一 str，click 转 int，缺失填 0
    df = pd.read_csv(file_path, nrows=100000, header=0, low_memory=False, dtype=str)
    df['click'] = pd.to_numeric(df['click'], errors='coerce').fillna(0).astype(np.int64)

    # 1.2 随机切分 80/20（数据全在同一小时，时间切分无意义）
    rng = np.random.RandomState(random_state)
    n = len(df)
    perm = rng.permutation(n)
    n_train = int(n * 0.8)
    train_df = df.iloc[perm[:n_train]].reset_index(drop=True).copy()
    test_df = df.iloc[perm[n_train:]].reset_index(drop=True).copy()

    # 1.3(a) 长尾截断：在训练集上统计频次，<10 归为 'Other'，规则套用到测试集
    site_freq = train_df['site_id'].value_counts()
    site_keep = set(site_freq[site_freq >= 10].index)
    app_freq = train_df['app_category'].value_counts()
    app_keep = set(app_freq[app_freq >= 10].index)

    def transform(d):
        d = d.copy()
        # (b) 小时提取：YYMMDDHH 取最后两位
        d['hour_bucket'] = d['hour'].astype(str).str[-2:].astype(int)
        d['site_id_group'] = d['site_id'].where(d['site_id'].isin(site_keep), 'Other')
        d['app_category_group'] = d['app_category'].where(d['app_category'].isin(app_keep), 'Other')
        return d

    train_t = transform(train_df)
    test_t = transform(test_df)

    # (c) 特征拼接与确定性哈希
    def to_states(d):
        h = d['hour_bucket'].values
        b = d['banner_pos'].values
        s = d['site_id_group'].values
        a = d['app_category_group'].values
        return np.array([_hash_state(hi, bi, si, ai)
                         for hi, bi, si, ai in zip(h, b, s, a)], dtype=np.int64)

    train_states = to_states(train_t)
    test_states = to_states(test_t)
    train_clicks = train_t['click'].values.astype(np.int64)
    test_clicks = test_t['click'].values.astype(np.int64)

    global_ctr = float(df['click'].mean())

    # Rule Baseline：训练集每个 banner_pos 的历史 CTR（仅在训练集计算，禁止数据泄露）
    train_banner_pos_ctr = train_df.groupby('banner_pos')['click'].mean().to_dict()
    test_banner_pos = test_df['banner_pos'].values

    return (train_states, train_clicks, test_states, test_clicks, global_ctr,
            train_banner_pos_ctr, test_banner_pos)
