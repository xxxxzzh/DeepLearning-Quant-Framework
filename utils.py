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