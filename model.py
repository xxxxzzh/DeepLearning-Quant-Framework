import torch
import torch.nn as nn

class TransformerAlpha(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, n_heads=4, num_layers=2):
        super(TransformerAlpha, self).__init__()
        self.embedding = nn.Linear(feature_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :]) # 只取序列最后一天
    

import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    """
    量化交易标准数据集封装类
    负责将 Numpy 数组转化为 PyTorch 认识的 Tensor，并喂给 Transformer
    """
    def __init__(self, features, targets):
        # 将传入的 8 维特征和目标收益率转换为 PyTorch 的张量格式
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        # 返回数据集的总行数
        return len(self.features)

    def __getitem__(self, idx):
        # Transformer 模型通常需要三维输入：(Batch_Size, Sequence_Length, Feature_Dim)
        # 你的特征是 [8]，我们用 unsqueeze(0) 把它变成 [1, 8]，代表序列长度为 1
        x = self.features[idx].unsqueeze(0)
        y = self.targets[idx]
        return x, y
    