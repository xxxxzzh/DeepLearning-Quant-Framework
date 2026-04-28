import pandas as pd
import pandas_ta as ta
import numpy as np

def process_panel_data(file_path="data/hs300_panel_raw.csv"):
    print(" 正在加载原始面板数据...")
    df = pd.read_csv(file_path)
    
    # 1. 翻译列名
    rename_map = {
        '日期': 'date', '开盘': 'open', '收盘': 'close', 
        '最高': 'high', '最低': 'low', '成交量': 'volume',
        '涨跌幅': 'pct_chg'
    }
    df.rename(columns=rename_map, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    
    # 排序：先按股票代码排，再按时间排
    df.sort_values(['ticker', 'date'], inplace=True)
    
    print(f" 加载成功，共 {len(df['ticker'].unique())} 只股票，总行数: {len(df)}")

    # ---------------------------------------------------------
    #  第一刀：按股票隔离 (Time-Series Features)
    # ---------------------------------------------------------
    print(" 正在逐只股票计算技术指标 (MA, RSI, MACD)...")
    
    processed_groups = []
    
    # 【终极防弹写法】：放弃诡异的 apply，直接用 for 循环 + concat
    # 这样 Pandas 绝对不敢偷偷删掉我们的 ticker 列
    for ticker, group in df.groupby('ticker'):
        group = group.copy()  # 深度复制，防止警告
        if len(group) < 35: 
            continue
            
        group['ma5'] = ta.sma(group['close'], length=5)
        group['rsi'] = ta.rsi(group['close'], length=14)
        group['atr'] = ta.atr(group['high'], group['low'], group['close'], length=14)
        
        macd_df = ta.macd(group['close'], fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            group['macd'] = macd_df.iloc[:, 0]
        else:
            group['macd'] = 0.0
            
        group['target_return'] = group['close'].pct_change().shift(-1)
        
        # 把处理好的单只股票装进列表
        processed_groups.append(group)

    # 像搭积木一样，把所有股票重新拼成一个完整的大表格
    df = pd.concat(processed_groups, ignore_index=True)
    
    # 砍掉开头因为计算 MACD 产生的 NaN
    df.dropna(subset=['ma5', 'rsi', 'macd', 'atr', 'target_return'], inplace=True)

    # ---------------------------------------------------------
    #  第二刀：按日期切片，做截面标准化 (Cross-Sectional Normalization)
    # ---------------------------------------------------------
    print(" 正在执行核心逻辑：逐日截面标准化 (Cross-Sectional Z-Score)...")
    
    features_to_norm = ['ma5', 'rsi', 'atr', 'macd', 'volume']
    
    # 用 transform 替代 apply，绝对安全
    for feat in features_to_norm:
        df[f'{feat}_norm'] = df.groupby('date')[feat].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 1e-6 else 0.0
        )

    # ---------------------------------------------------------
    #  收尾：对齐与导出
    # ---------------------------------------------------------
    print(" 正在对齐时间序列并剔除残缺数据...")
    
    # 此时 date 和 ticker 绝对安全，不会再报错了
    final_cols = ['date', 'ticker', 'target_return'] + [f'{f}_norm' for f in features_to_norm]
    final_df = df[final_cols].copy()
    
    # 清洗无穷大的脏数据
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.dropna(inplace=True)

    save_path = "data/hs300_panel_features.csv"
    final_df.to_csv(save_path, index=False)
    
    print(f" 截面特征工厂竣工！标准面板数据已保存至: {save_path}")
    print(f" 最终特征矩阵规模: {final_df.shape}")
    
    return final_df

if __name__ == "__main__":
    process_panel_data()