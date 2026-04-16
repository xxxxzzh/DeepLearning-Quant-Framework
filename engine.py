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
            # 1. 基础信号：模型预测值（标准化处理，让它落在 0-1 之间）
            # 假设 pred 是你的预测收益率
            temp_df['model_score'] = np.where(temp_df['pred'] > temp_df['pred'].median(), 0.6, 0)
            
            # 2. 辅助评分：MACD 趋势评分
            if 'macd_line' in temp_df.columns and 'signal_line' in temp_df.columns:
                # 金叉给 0.4 分，死叉给 0 分
                temp_df['trend_score'] = np.where(temp_df['macd_line'] > temp_df['signal_line'], 0.4, 0)
            else:
                temp_df['trend_score'] = 0
                
            # 3. 最终信号 = 模型分(60%) + 趋势分(40%)
            # 这样即便模型看错，如果有趋势保护，仓位也会被修正
            temp_df['signal'] = temp_df['model_score'] + temp_df['trend_score']
            # 4. RSI 极端行情熔断（风控开关）
            # 如果 RSI > 80，说明极度超买，即便评分再高也强行把仓位降到 0.2（试探性仓位）
            if 'rsi_14' in temp_df.columns:
                temp_df.loc[temp_df['rsi_14'] > 80, 'signal'] = 0.2
                
        # 2. 计算收益与成本
        temp_df['pos_diff'] = temp_df['signal'].diff().abs().fillna(0)
        temp_df['strat_ret_raw'] = temp_df['signal'].shift(1) * temp_df['actual']
        temp_df['trade_cost'] = temp_df['pos_diff'] * self.cost
        temp_df['strat_ret_net'] = temp_df['strat_ret_raw'] - temp_df['trade_cost']
        
        # 3. 计算累计净值
        temp_df['cum_strategy'] = (1 + temp_df['strat_ret_net'].fillna(0)).cumprod()
        temp_df['cum_market'] = (1 + temp_df['actual']).cumprod()
        
        return temp_df