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
            # --- 1. 因子分标准化 (Normalization) ---
            # LSTM 预测分：假设预测 0.5% 为满分 1.0
            score_lstm = (temp_df['pred'] / 0.005).clip(-1, 1)
    
            # 趋势分 (MACD)：金叉为 1，死叉为 -1
            score_macd = np.where(temp_df['macd_line'] > temp_df['signal_line'], 1.0, -1.0)
    
            # 均值回归分 (RSI)：将 0~100 映射到 1 到 -1
            # 50 是中点 (0分)，80 以上是负分，20 以下是正分
            score_rsi = (50 - temp_df['rsi_14']) / 30 
            score_rsi = score_rsi.clip(-1, 1)

            # --- 2. 权重分配 (按你刚才的想法) ---
            # 给 LSTM 较高权重 (比如 0.6)，给趋势和 RSI 各 0.2
            w_lstm, w_macd, w_rsi = 0.6, 0.2, 0.2
    
            temp_df['total_score'] = (
                score_lstm * w_lstm + 
                score_macd * w_macd + 
                score_rsi * w_rsi
            )

            # --- 3. 加权后的缓冲区逻辑 (Hysteresis) ---
            # 只有总分超过 0.5 才买入，低于 -0.2 就清仓
            buy_threshold = 0.5
            sell_threshold = -0.2
    
            temp_df['signal'] = 0.0
            for i in range(1, len(temp_df)):
                prev_sig = temp_df.loc[temp_df.index[i-1], 'signal']
                current_score = temp_df.loc[temp_df.index[i], 'total_score']
        
                if current_score > buy_threshold:
                    temp_df.loc[temp_df.index[i], 'signal'] = 0.8 # 固定 0.8 仓位
                elif current_score < sell_threshold:
                    temp_df.loc[temp_df.index[i], 'signal'] = 0.0
                else:
                    temp_df.loc[temp_df.index[i], 'signal'] = prev_sig

           
            # --- 调试代码开始 ---
            triggered_count = (temp_df['rsi_14'] > 0.8).sum()
            actual_signals = temp_df['signal'].unique()
            print(f"DEBUG: RSI>0.8的天数: {triggered_count} | 现在的信号种类: {actual_signals}")
            # --- 调试代码结束 ---

        # 2. 计算收益与成本
        self.cost = 0.001  # 千分之一的手续费（买卖双向合计）
        temp_df['pos_diff'] = temp_df['signal'].diff().abs().fillna(0)
        temp_df['trade_cost'] = temp_df['pos_diff'] * self.cost
        temp_df['strat_ret_raw'] = temp_df['signal'].shift(1) * temp_df['actual']
        temp_df['strat_ret_net'] = temp_df['strat_ret_raw'] - temp_df['trade_cost']
        
        # 3. 计算累计净值
        temp_df['cum_strategy'] = (1 + temp_df['strat_ret_net'].fillna(0)).cumprod()
        temp_df['cum_market'] = (1 + temp_df['actual']).cumprod()
        
        return temp_df