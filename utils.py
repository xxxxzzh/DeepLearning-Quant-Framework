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

import pandas as pd
import numpy as np

def calculate_ic_series(df):
    """
    终极修复版：强制单序列数据输出有效 IC
    """
    # 确保数据按日期排序
    df = df.sort_values('date')
    
    # 判断是否为单序列（每天只有一行）
    if df.groupby('date').size().max() == 1:
        # 使用 20 天滚动窗口计算相关性
        # min_periods=5 表示只要有5天数据就开始计算，不非得等够20天
        ic_series = df['actual'].rolling(window=20, min_periods=5).corr(df['pred'])
        
        # 核心：去掉开头的 NaN，并处理可能出现的常数列导致的 null
        ic_series = ic_series.replace([np.inf, -np.inf], np.nan).dropna()
        return ic_series
    else:
        # 多序列截面逻辑
        return df.groupby('date').apply(lambda x: x['actual'].corr(x['pred'], method='spearman')).dropna()


