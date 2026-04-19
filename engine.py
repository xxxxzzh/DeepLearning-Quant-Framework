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
            # --- 1. 先初始化一个全是 0 的 signal 列 (解决报错的关键) ---
            temp_df['signal'] = 0.0
    
            # --- 2. 计算理想状态的分数 ---
            confidence_scale = 100.0
            raw_pred_score = (temp_df['pred'] * confidence_scale).clip(0, 1)
    
            # 确保列名正确，根据你的数据检查是否叫 'macd_line'
            trend_score = np.where(temp_df['macd_line'] > temp_df['signal_line'], 0.4, 0)
            ideal_signal = (raw_pred_score + trend_score).clip(0, 1)

            # --- 3. 缓冲区逻辑 ---
            buy_threshold = 0.005
            sell_threshold = -0.002
            
            # 为了保证效率，我们先生成一个临时的 shift 序列
            # 注意：在回测引擎初次启动时，第一天的 prev_signal 默认为 0
    
            # 我们用一个循环或者更简单的方式处理：
            # 这里我们采用一种“向量化”的折中写法：
            for i in range(1, len(temp_df)):
                prev_sig = temp_df.loc[temp_df.index[i-1], 'signal']
                current_pred = temp_df.loc[temp_df.index[i], 'pred']
        
                if current_pred > buy_threshold:
                    # 买入时固定仓位，不要每天微调 (比如固定 0.8)
                    # 这样只要不卖出，中间就不产生任何调仓手续费
                    temp_df.loc[temp_df.index[i], 'signal'] = 0.8 
                elif current_pred < sell_threshold:
                    temp_df.loc[temp_df.index[i], 'signal'] = 0.0
                else:
                    # 关键：在这里，保持和昨天一模一样！
                    temp_df.loc[temp_df.index[i], 'signal'] = prev_sig

            # 3. RSI 风险熔断 (依然保留)
            rsi_max = temp_df['rsi_14'].max()
            threshold = 0.8 if rsi_max <= 1.0 else 80
            temp_df.loc[temp_df['rsi_14'] > threshold, 'signal'] = 0.0

       
            # 4. 【新增】信号防抖逻辑（Throttling）：省去不必要的手续费
            # 只有当新信号和旧信号差别大于 5% 时才动，否则保持原样
            temp_df['prev_signal'] = temp_df['signal'].shift(1).fillna(0)
            temp_df['signal'] = np.where(
                abs(temp_df['signal'] - temp_df['prev_signal']) < 0.1, 
                temp_df['prev_signal'], 
                temp_df['signal']
            )
    
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