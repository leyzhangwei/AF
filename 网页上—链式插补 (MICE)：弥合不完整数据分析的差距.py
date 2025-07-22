# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 09:49:50 2025

@author: admin
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 生成合成数据集
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# 转换为 DataFrame，以便于操作
df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
df['Target'] = y

# 显示前几行
df.head()

# I引入缺失值
np.random.seed(42)
df_missing = df.mask(np.random.random(df.shape) < 0.1)

# 显示前几行以验证缺失值
df_missing.head()

# 初始化 MICE 计算器
mice_imputer = IterativeImputer(max_iter=5, random_state=42)

# 拟合和转换数据集以填补缺失值
df_imputed = mice_imputer.fit_transform(df_missing)

# 将拟合数据转换回 pandas DataFrame
df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
df_imputed.head()

# 计算每个特征的 RMSE
rmse = np.sqrt(mean_squared_error(df, df_imputed, multioutput='raw_values'))

# 打印每个特征的均方根误差
print(f'RMSE for each feature: {rmse}')

# 选择要绘制的特征
feature_to_plot = 'Feature_0'

# 绘制原始分布图和处理后的分布图
plt.figure(figsize=(10, 6))
sns.kdeplot(df[feature_to_plot], label='Original', color='green', linestyle="--")
sns.kdeplot(df_imputed[feature_to_plot], label='Imputed', color='red', linestyle="-")
plt.legend()
plt.title(f'Distribution of Original vs. Imputed Values for {feature_to_plot}')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()