import numpy as np

def calculate_metrics(df, name):
    # 强制转换为 numpy 数组，规避所有 Pandas 维度报错
    cum_strategy = df['cum_strategy'].values
    net_returns = df['strat_ret_net'].fillna(0).values
    
    # 总收益
    total_ret = cum_strategy[-1] - 1
    
    # 夏普比率 (年化)
    std = net_returns.std()
    sharpe = (net_returns.mean() / std * np.sqrt(252)) if std != 0 else 0
    
    # 最大回撤 (MDD)
    running_max = np.maximum.accumulate(cum_strategy)
    drawdowns = (cum_strategy - running_max) / running_max
    max_dd = np.abs(np.min(drawdowns))
    
    return {
        '策略名称': name, 
        '总收益': f"{total_ret:.2%}", 
        '夏普比率': f"{sharpe:.2f}", 
        '最大回撤': f"{max_dd:.2%}"
    }

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def calculate_rank_ic(y_true, y_pred):
    """
    计算 Rank IC (Spearman 相关系数)
    y_true: 真实收益率
    y_pred: 模型预测收益率
    """
    # 移除空值
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return 0.0
    
    # spearmanr 返回 (correlation, p-value)
    correlation, _ = spearmanr(y_true[mask], y_pred[mask])
    return correlation