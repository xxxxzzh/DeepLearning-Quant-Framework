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
from utils import calculate_rank_ic,calculate_ic_series
def run_quant_system():
    print(" 量化交易系统 1.0 (Transformer 增强版) 正在启动...")
    
    # --- 第一步：加载原始数据 (含特征，不含预测) ---
    # 假设你的原始特征数据在 data.csv 里，或者你从结果文件里读取特征
    try:
        raw_df = pd.read_csv('enhanced_performance_results.csv') 
        raw_df['date'] = pd.to_datetime(raw_df['date'])
        # 截取 2023 年以后的数据
        results_df = raw_df[raw_df['date'] >= '2023-01-01'].copy().reset_index(drop=True)
    except FileNotFoundError:
        print(" 错误：找不到数据文件")
        return

    # --- 第二步：【新增】使用 Transformer 生成预测 (这一步是 Month 4 的核心) ---
    print("--- 正在调用 Transformer 模型进行信号推理 ---")
    
    # 1. 实例化模型 (假设特征维度是 6)
    feature_dim = 6 
    model = TransformerAlpha(feature_dim=feature_dim)
    
    # 2. 加载你之前练好的权重 (如果你已经训练好了)
    # model.load_state_dict(torch.load('transformer_best.pt'))
    # model.eval() 
    
    # 3. 这里为了演示，我们假设 Transformer 产生的预测值更新了 results_df['pred']
    # 如果你现在只想先跑通流程，可以先保留 CSV 里的 pred，
    # 等你把模型训练代码写好后再替换它。
    
    # --- 第三步：数据检查 (保持你原来的代码) ---
    print(f"--- 开启近3年实战回测，样本量: {len(results_df)} 天 ---")
    results_df = results_df.rename(columns={'return': 'actual'}) 
    
    # --- 第四步：启动引擎 (使用重构后的 QuantBacktester) ---
    tester = QuantBacktester(results_df)

    # 运行策略
    # 注意：现在的 relative 逻辑会自动去 engine 里找 StrategyManager 进行加权评分
    res_abs = tester.run_backtest(signal_type='absolute', threshold=0)
    res_rel = tester.run_backtest(signal_type='relative')

    # --- 第五步：指标计算与打印 ---
 
    #  计算每日 IC 序列
    ic_series = calculate_ic_series(results_df)

    # 计算核心指标
    from utils import calculate_rank_ic
    ic_value = calculate_rank_ic(results_df['actual'], results_df['pred'])
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    icir = mean_ic / std_ic if std_ic != 0 else 0

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