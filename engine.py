import pandas as pd
import numpy as np

class QuantBacktester:
    def __init__(self, df, fees=0.001, slippage=0.0005):
        self.df = df.copy()
        if 'close' in self.df.columns:
            # 计算 5 日均线
            self.df['ma5'] = self.df['close'].rolling(window=5).mean()
            # 计算偏离度 (神经网络和量化模型更喜欢的平稳特征)
            self.df['ma5_bias'] = (self.df['close'] / self.df['ma5']) - 1
            # 处理因为滚动计算产生的 NaN (初始几天没有均线)
            self.df = self.df.fillna(0)
        # 确保日期格式正确
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
        self.cost = fees + slippage
        
    def run_backtest(self, signal_type='absolute', threshold=0):
        """
        核心回测函数
        signal_type: 'absolute' (阈值法) 或 'relative' (排名法)
        """
        temp_df = self.df.copy()
        
        # 1. 信号生成
        if signal_type == 'absolute':
            # 基础逻辑：预测大于阈值
            condition = (temp_df['pred'] > threshold)
            
            # 进阶过滤：只有当 RSI 不处于高位超买区时才买入
            if 'rsi_14' in temp_df.columns:
                condition = condition & (temp_df['rsi_14'] < 75)
            
            temp_df['signal'] = np.where(condition, 1, 0)

        elif signal_type == 'relative':
            # 基础逻辑：预测优于中位数
            condition = (temp_df['pred'] > temp_df['pred'].median())
            
            # 进阶过滤：MACD 处于上升趋势（macd > signal_line）
            if 'macd_line' in temp_df.columns and 'signal_line' in temp_df.columns:
                 condition = condition & (temp_df['macd_line'] > temp_df['signal_line'])
            
            temp_df['signal'] = np.where(condition, 1, 0)
            
        # 2. 计算收益与成本
        temp_df['pos_diff'] = temp_df['signal'].diff().abs().fillna(0)
        temp_df['strat_ret_raw'] = temp_df['signal'].shift(1) * temp_df['actual']
        temp_df['trade_cost'] = temp_df['pos_diff'] * self.cost
        temp_df['strat_ret_net'] = temp_df['strat_ret_raw'] - temp_df['trade_cost']
        
        # 3. 计算累计净值
        temp_df['cum_strategy'] = (1 + temp_df['strat_ret_net'].fillna(0)).cumprod()
        temp_df['cum_market'] = (1 + temp_df['actual']).cumprod()
        
        return temp_df