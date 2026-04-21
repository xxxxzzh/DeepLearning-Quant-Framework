# 1. 导入
import os
import sys

# 1. 允许重复库运行
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. 【核心修复】强制把 torch 的 lib 目录加入 Windows 搜索路径
import torch
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib_path):
    os.add_dll_directory(torch_lib_path)

# 3. 现在的导入顺序
from engine import QuantBacktester
from model import TransformerAlpha
from utils import calculate_metrics
import pandas as pd
import numpy as np
from utils import calculate_rank_ic,calculate_ic_series
import pandas_ta as ta  
from model import TransformerAlpha, StockDataset  
from torch.utils.data import DataLoader       
import random

def set_seed(seed=42):
    """固定所有的随机种子，确保模型结果 100% 可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 如果你用了 GPU，还需要加上下面两行
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 强制让 PyTorch 的底层卷积运算具备确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 在程序一开始就调用它！
set_seed(42)
def get_cleaned_data(df):
    """
    将原始行情转化为深度学习特征
    """
    df.columns = [c.lower() for c in df.columns]
    
    # --- 因子注入 ---
    df.ta.macd(append=True)   # 趋势因子
    df.ta.rsi(append=True)    # 强弱因子
    df.ta.atr(append=True)    # 波动率因子
    df.ta.sma(length=5, append=True)   # 短期均线
    df.ta.sma(length=20, append=True)  # 中期均线
    
    # 填充技术指标开头的空值
    df = df.fillna(0)
    
    # --- 最终选定的 8 维特征 ---
    # 注意：运行一次后如果报错，请 print(df.columns) 检查确切的大小写
    feature_cols = [
        'close', 'volume', 'rsi_14', 
        'macd_12_26_9', 'macdh_12_26_9', 
        'atr_14', 'sma_5', 'sma_20'
    ]
    
    return df[feature_cols], df['close'] # 返回特征和用于回测的价格

import torch.nn as nn

def train_model(model, train_loader, epochs=10, lr=0.001):
    model.train() # 切换到训练模式
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() # 量化预测通常使用均方误差损失
    
    print(f"开始训练模型... 计划运行 {epochs} 轮")
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            # 1. 清空梯度 (面试必考点！)
            optimizer.zero_grad()
            
            # 2. 前向传播
            outputs = model(batch_x)
            
            # 3. 计算损失
            loss = criterion(outputs.squeeze(), batch_y.float())
            
            # 4. 反向传播与优化
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.6f}")
    print("模型训练完成！")

def run_quant_system():
    print("量化交易系统 1.0 (Transformer 增强版) 正在启动...")
    
    # --- 第一步：加载原始数据 ---
    try:
        raw_df = pd.read_csv('enhanced_performance_results.csv') 
        raw_df['date'] = pd.to_datetime(raw_df['date'])
        
        # 无情剔除重复的列名 (解决 Pandas 报错和 9维 问题)
        raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()]
        
        # 填平所有基础空值 (防止标签 Return 中藏有 NaN)
        raw_df = raw_df.fillna(0)
        
    except FileNotFoundError:
        print(" 错误：找不到数据文件")
        return

    # --- 第二步：特征工程与标准化 ---
    tester_temp = QuantBacktester(raw_df)
    features = tester_temp.get_feature_matrix() 
    features = features.astype(np.float64) 
    
    # 安全的 Z-Score 标准化 (彻底剿灭导致 Loss 为 nan 的隐患)
    # 使用 np.nanmean 忽略潜在的 nan，并用 nan_to_num 将异常值强行归零
    features_norm = (features - np.nanmean(features, axis=0)) / (np.nanstd(features, axis=0) + 1e-9)
    features_norm = np.nan_to_num(features_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 在 main.py 准备 targets 的时候，正确的做法是预测 T+1 的收益：
    raw_df['target'] = raw_df['return'].shift(-1)
    targets = np.nan_to_num(raw_df['target'].values, nan=0.0)

    # 强制裁剪，确保特征矩阵和标签向量的长度完全一致
    min_len = min(len(features_norm), len(targets))
    features_norm = features_norm[:min_len]
    targets = targets[:min_len]
    # --- 第三步：模型训练 ---
    print("--- 正在进行特征磨合训练 (Training Loop) ---")
    # 动态获取特征矩阵的列数（不再写死 8）
    feature_dim = features_norm.shape[1] 
    print(f" 模型入口维度已自适应调整为: {feature_dim} 维")
    
    # 实例化模型（注意：确认你之前的参数名是 feature_dim 还是 input_dim）
    model = TransformerAlpha(feature_dim=feature_dim) 
    
    # 准备 Dataset 和 DataLoader
    dataset = StockDataset(features_norm, targets)
    # 注意：训练时建议 shuffle=True 增加泛化能力
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True) 

    # 【正式激活训练】
    train_model(model, train_loader, epochs=10) #  10 轮

    # --- 第四步：实时推理 (用练好的模型生成新信号) ---
    print("--- 正在根据 8 维特征生成最新预测信号 ---")
    model.eval() # 切换到评估模式，关闭 Dropout
    
    # 为了保持顺序，我们用一个新的 shuffle=False 的 Loader 来拿预测值
    eval_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    with torch.no_grad():
        for x, _ in eval_loader:
            pred = model(x)
            all_preds.append(pred.squeeze().detach().cpu().numpy())
    
    # 合并预测结果 (处理成一维数组)
    new_predictions = np.concatenate([p.flatten() if p.ndim > 0 else [p] for p in all_preds])
    
    # 确保长度对齐 (如果有偏移，取最后的部分)
    # 假设 dataset 的长度和 raw_df 对齐
    raw_df['pred'] = new_predictions
    

    # --- 第五步：截取 2023 年后的数据进行实战回测 ---
    results_df = raw_df[raw_df['date'] >= '2023-01-01'].copy().reset_index(drop=True)
    results_df = results_df.rename(columns={'return': 'actual'}) 
    
    print(f"--- 开启近3年实战回测，样本量: {len(results_df)} 天 ---")
    
    # --- 第六步：启动回测引擎 (使用新预测值) ---
    tester = QuantBacktester(results_df)
    res_abs = tester.run_backtest(signal_type='absolute', threshold=0)
    res_rel = tester.run_backtest(signal_type='relative')


    # --- 第五步：指标计算与打印 ---
    # 1. 计算总体 Rank IC (Overall)
    ic_value = results_df['actual'].corr(results_df['pred'], method='spearman')

    # 2. 计算 IC 序列 (Rolling)
    ic_series = calculate_ic_series(results_df)

    # 3. 计算均值和稳定性
    if not ic_series.empty:
        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        icir = mean_ic / std_ic if (std_ic != 0 and not np.isnan(std_ic)) else 0
    else:
        mean_ic = 0.0
        icir = 0.0


    # 4. 运行两种策略对比
    # 策略 A: 绝对阈值法（保守）
    res_abs = tester.run_backtest(signal_type='absolute', threshold=0)
    
    # 策略 B: 相对排名法（激进 - 我们昨天的重大进化！）
    res_rel = tester.run_backtest(signal_type='relative')

    # 5. 计算并显示指标
    metrics_a = calculate_metrics(res_abs, "保守型 (Absolute)")
    metrics_b = calculate_metrics(res_rel, "激进型 (Relative)")

    # 6. 打印精美的对比报告
    total_cost = res_rel['trade_cost'].sum()

    if np.isnan(mean_ic):
        print(f" 预测能力 (Rank IC - Overall): {ic_value:.4f}")
        print(f" 预测能力 (Rank IC): {ic_value:.4f}")
        print(f" 预测稳定性 (ICIR): 择时策略建议参考总 Rank IC")
    else:
        print(f" 预测能力 (Mean Rank IC): {mean_ic:.4f}")

    print(f" 预测稳定性 (ICIR): {icir:.4f}")
    print(f" 激进型策略总交易成本: {total_cost:.2%}")
    print("\n" + "="*40)
    print("         策略表现对比报告")
    print(f" 预测能力 (Rank IC): {ic_value:.4f}")
    print(f" 预测能力 (Mean Rank IC): {mean_ic:.4f}")
    print(f" 预测稳定性 (ICIR): {icir:.4f}")
    print("="*40)
    for m in [metrics_a, metrics_b]:
        print(f"策略: {m['策略名称']}")
        print(f"  - 总收益:   {m['总收益']}")
        print(f"  - 夏普比率: {m['夏普比率']}")
        print(f"  - 最大回撤: {m['最大回撤']}")
        print("-" * 20)
    print("="*40)

if __name__ == "__main__":
    run_quant_system()