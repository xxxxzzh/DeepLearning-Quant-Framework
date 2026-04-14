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

def calculate_rank_ic(actual, pred):
    """
    计算 Rank IC (Spearman 相关系数)
    actual: 真实收益率
    pred: 模型预测收益率
    """
    # 移除空值
    mask = ~np.isnan(actual) & ~np.isnan(pred)
    if not np.any(mask):
        return 0.0
    
    # spearmanr 返回 (correlation, p-value)
    correlation, _ = spearmanr(actual[mask], pred[mask])
    return correlation

def calculate_ic_series(df, target_col='actual', pred_col='pred'):
    """
    按日期计算每日 Rank IC,并过滤无效值
    """
    # 1. 确保日期是时间格式
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. 按天计算相关系数
    # 我们加一个检查：只有当当天的样本数 > 1 时才计算
    ic_series = df.groupby('date').apply(
        lambda x: x[target_col].corr(x[pred_col], method='spearman') if len(x) > 1 else np.nan
    )
    
    # 3. 剔除计算失败的 nan 值，否则平均值也会变成 nan
    return ic_series.dropna()