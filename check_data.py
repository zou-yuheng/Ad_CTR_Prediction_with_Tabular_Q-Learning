# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 15:14:54 2026

@author: xingh
"""

import pandas as pd
import hashlib
import numpy as np

# ================= 1. 数据加载 =================
print("⏳ 正在读取前 10 万条数据...")
file_path = 'train.csv'
columns = [
    'id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 
    'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 
    'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'
]

# 强制指定数据类型，避免 mixed types 报错
dtype_spec = {
    'id': str,
    'click': float,      # 强制转为浮点数，方便后续求均值
    'hour': str,         # hour 字段可能包含非数字字符，先当字符串读
    'C1': str,
    'banner_pos': str,
    'site_id': str,
    'site_domain': str,
    'site_category': str,
    'app_id': str,
    'app_domain': str,
    'app_category': str,
    'device_id': str,
    'device_ip': str,
    'device_model': str,
    'device_type': str,
    'device_conn_type': str,
    'C14': str,
    'C15': str,
    'C16': str,
    'C17': str,
    'C18': str,
    'C19': str,
    'C20': str,
    'C21': str
}

df = pd.read_csv(
    file_path, 
    nrows=100000, 
    names=columns, 
    header=0, 
    dtype=dtype_spec,    # 传入类型字典
    na_values=['', ' ', 'null', 'NULL', 'None'],  # 把常见的空值标记统一识别为 NaN
    low_memory=False     # 关闭分块读取警告
)

# 额外保险：将 click 列中的 NaN 填充为 0，并转为整型
df['click'] = df['click'].fillna(0).astype(int)

# ================= 2. 核心指标侦察 =================
print("\n" + "="*50)
print("📊 1. 全局基准与时间漂移检查")
print("="*50)

# 划分训练集和测试集 (前8万，后2万)
train_df = df.iloc[:80000]
test_df = df.iloc[80000:]

print(f"全局平均 CTR: {df['click'].mean():.4f}")
print(f"训练集平均 CTR: {train_df['click'].mean():.4f} | 时间范围: {train_df['hour'].min()} ~ {train_df['hour'].max()}")
print(f"测试集平均 CTR: {test_df['click'].mean():.4f} | 时间范围: {test_df['hour'].min()} ~ {test_df['hour'].max()}")
print("💡 导师提示: 如果训练集和测试集的 CTR 或时间范围差异巨大，说明存在严重的时间漂移。")

# ================= 3. 特征基数与长尾分布 =================
print("\n" + "="*50)
print("📊 2. 特征基数与长尾分布")
print("="*50)

for col in ['site_id', 'app_category', 'banner_pos']:
    nunique = df[col].nunique()
    print(f"特征 [{col}] 唯一值数量: {nunique}")

# 重点检查 site_id 的长尾
site_counts = df['site_id'].value_counts()
top10_ratio = site_counts.head(10).sum() / len(df)
low_freq_count = (site_counts < 10).sum()

print(f"\n[site_id] 频次 Top 10 占据总样本比例: {top10_ratio:.2%}")
print(f"[site_id] 频次小于 10 的类别数量: {low_freq_count} / {len(site_counts)}")
print("💡 导师提示: 如果 Top10 占比极高且低频类别极多，说明长尾截断（归为 Other）非常必要。")

# ================= 4. 联合状态空间与哈希验证 =================
print("\n" + "="*50)
print("📊 3. 联合状态空间与哈希冲突验证")
print("="*50)

# 模拟特征工程：提取小时，长尾处理
def preprocess_for_hash(df):
    temp = df.copy()
    temp['hour_bucket'] = temp['hour'].astype(str).str[-2:].astype(int)
    # 频次 < 10 归为 Other
    site_freq = temp['site_id'].value_counts()
    temp['site_id_group'] = temp['site_id'].apply(lambda x: x if site_freq.get(x, 0) >= 10 else 'Other')
    
    app_freq = temp['app_category'].value_counts()
    temp['app_category_group'] = temp['app_category'].apply(lambda x: x if app_freq.get(x, 0) >= 10 else 'Other')
    return temp

train_processed = preprocess_for_hash(train_df)

# 拼接特征字符串
state_str_series = (
    train_processed['hour_bucket'].astype(str) + '|' +
    train_processed['banner_pos'].astype(str) + '|' +
    train_processed['site_id_group'].astype(str) + '|' +
    train_processed['app_category_group'].astype(str)
)

# 使用确定性哈希 (MD5)
hash_buckets = 2048
state_ids = state_str_series.apply(lambda x: int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16) % hash_buckets)

# 统计哈希桶的访问情况
bucket_counts = state_ids.value_counts()
active_buckets = len(bucket_counts)
low_sample_buckets = (bucket_counts < 5).sum()

print(f"设定的哈希桶数量: {hash_buckets}")
print(f"训练集实际激活的桶数量: {active_buckets}")
print(f"样本数 < 5 的桶数量: {low_sample_buckets} (占比: {low_sample_buckets/active_buckets:.2%})")
print("💡 导师提示: 如果激活桶极少，说明状态空间被压缩得很厉害；如果低样本桶占比过高，Top-k 排序时必须做平滑处理。")

# 检查测试集落入训练集哈希桶的比例
test_processed = preprocess_for_hash(test_df)
test_state_str = (
    test_processed['hour_bucket'].astype(str) + '|' +
    test_processed['banner_pos'].astype(str) + '|' +
    test_processed['site_id_group'].astype(str) + '|' +
    test_processed['app_category_group'].astype(str)
)
test_state_ids = test_state_str.apply(lambda x: int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16) % hash_buckets)

unseen_ratio = (~test_state_ids.isin(state_ids)).mean()
print(f"\n测试集中未见过状态(Out-of-Vocabulary)的比例: {unseen_ratio:.2%}")
print("💡 导师提示: 如果该比例 > 10%，说明时间漂移严重，模型在测试集上会大量盲猜。")

print("\n🎉 EDA 侦察完成！请查看上方结果。")