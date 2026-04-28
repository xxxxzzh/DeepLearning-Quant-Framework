import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class CrossSectionalDataset(Dataset):
    def __init__(self, file_path="data/hs300_panel_features.csv"):
        print(" 正在将 Pandas 面板数据转化为 PyTorch 3D 张量...")
        self.df = pd.read_csv(file_path)
        
        # 动态提取所有的特征列 (排除掉不需要进神经网络的标记列)
        self.feature_cols = [c for c in self.df.columns if c not in ['date', 'ticker', 'target_return']]
        
        # 获取所有唯一的交易日，并按时间严格排序
        self.dates = sorted(self.df['date'].unique())
        print(f" 成功提取 {len(self.dates)} 个交易日的截面快照。")

    def __len__(self):
        # 数据集的总长度，就是交易日的总天数
        return len(self.dates)

    def __getitem__(self, idx):
        # 取出特定某一天的全部股票数据 (横向切片)
        target_date = self.dates[idx]
        day_data = self.df[self.df['date'] == target_date]
        
        # 提取特征 X (形状: [当天的股票数量, 特征维度])
        X = day_data[self.feature_cols].values
        
        # 提取目标 Y (形状: [当天的股票数量])
        Y = day_data['target_return'].values
        
        # 转换为 PyTorch 的 Tensor，并把那天的 ticker 名单带上，方便后期回测
        return {
            'date': target_date,
            'X': torch.tensor(X, dtype=torch.float32),
            'Y': torch.tensor(Y, dtype=torch.float32),
            'tickers': day_data['ticker'].values.tolist() 
        }

# =========================================
# 本地测试模块 (防呆检验)
# =========================================
if __name__ == "__main__":
    dataset = CrossSectionalDataset()
    
    # 极其关键：做截面模型时，batch_size 必须设为 1！
    # 因为每天在市交易的股票数量可能不一样 (如停牌)，强行设为 32 会导致 Tensor 形状不匹配而报错。
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 我们只取第一天的数据出来看看形状
    for batch in dataloader:
        print("\n" + "="*40)
        print(f" 交易日: {batch['date'][0]}")
        print(f" 股票池: {batch['tickers']}")
        print(f" 特征张量 X 形状: {batch['X'].shape} -> [Batch, 股票数量, 特征维度]")
        print(f" 目标张量 Y 形状: {batch['Y'].shape} -> [Batch, 股票数量]")
        print("="*40)
        break