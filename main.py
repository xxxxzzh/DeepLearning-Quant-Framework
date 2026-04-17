# 1. 导入你刚刚写的两个模块
from engine import QuantBacktester
from utils import calculate_metrics
import pandas as pd
from utils import calculate_rank_ic,calculate_ic_series
def run_quant_system():
    print(" 量化交易系统 1.0 正在启动...")
    
    # 2. 加载之前生成的预测结果
    try:
        results_df = pd.read_csv('enhanced_performance_results.csv')
        results_df = results_df.rename(columns={'return': 'actual'}) 
        # 在加载完 results_df 后，只看 2023 年以后的表现
        results_df['date'] = pd.to_datetime(results_df['date'])
        # 截取 2023 年以后的数据
        results_df = results_df[results_df['date'] >= '2023-01-01'].reset_index(drop=True)
        print(f"--- 开启近3年实战回测，样本量: {len(results_df)} 天 ---")
        tester = QuantBacktester(results_df)
        print(" 成功将 'return' 重命名为 'actual'")
        print("文件里的列名有：", results_df.columns.tolist())
        print(" 成功加载预测数据")
    except FileNotFoundError:
        print(" 错误：找不到数据文件，请确保 CSV 文件在同一目录下")
        return
    date_counts = results_df['date'].value_counts()
    print("\n--- 数据分布检查 ---")
    print(f"总行数: {len(results_df)}")
    print(f"不同日期数量: {len(date_counts)}")
    print(f"每组日期平均样本数: {date_counts.mean():.2f}")
    print("前 5 天的数据量样本:\n", date_counts.head())
    print(f"RSI 列的空值数量: {results_df['rsi_14'].isna().sum()}")

    ic_value = calculate_rank_ic(results_df['actual'], results_df['pred'])
    # 3. 初始化引擎
    tester = QuantBacktester(results_df)

    #  计算每日 IC 序列
    ic_series = calculate_ic_series(results_df)

    # 计算核心指标
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