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
        # 在 run_backtest 函数中修改 relative 分支
        elif signal_type == 'relative':
            # 1. 择时信号：不再用 median 排名，而是看预测值是否大于 0
            # 0.6 是基础分，代表模型看好
            temp_df['model_score'] = np.where(temp_df['pred'] > 0, 0.6, 0)
    
            # 2. 趋势评分：MACD 金叉
            if 'macd_line' in temp_df.columns:
                temp_df['trend_score'] = np.where(temp_df['macd_line'] > temp_df['signal_line'], 0.4, 0)
            else:
                temp_df['trend_score'] = 0
        
            temp_df['signal'] = temp_df['model_score'] + temp_df['trend_score']

            # 3. 修复 RSI 阈值
            # 关键：检查你的 RSI 是否被 MinMaxScaler 缩放到了 0-1
            rsi_max = temp_df['rsi_14'].max()
            threshold = 0.8 if rsi_max <= 1.0 else 80
    
            if 'rsi_14' in temp_df.columns:
                # 只要 RSI 超过阈值，强行降仓到 0.2
                temp_df.loc[temp_df['rsi_14'] > threshold, 'signal'] = 0.2

            # --- 调试代码开始 ---
            triggered_count = (temp_df['rsi_14'] > 0.8).sum()
            actual_signals = temp_df['signal'].unique()
            print(f"DEBUG: RSI>0.8的天数: {triggered_count} | 现在的信号种类: {actual_signals}")
            # --- 调试代码结束 ---

        # 2. 计算收益与成本
        temp_df['pos_diff'] = temp_df['signal'].diff().abs().fillna(0)
        temp_df['strat_ret_raw'] = temp_df['signal'].shift(1) * temp_df['actual']
        temp_df['trade_cost'] = temp_df['pos_diff'] * self.cost
        temp_df['strat_ret_net'] = temp_df['strat_ret_raw'] - temp_df['trade_cost']
        
        # 3. 计算累计净值
        temp_df['cum_strategy'] = (1 + temp_df['strat_ret_net'].fillna(0)).cumprod()
        temp_df['cum_market'] = (1 + temp_df['actual']).cumprod()
        
        return temp_df