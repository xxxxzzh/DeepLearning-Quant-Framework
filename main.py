# 1. 导入
#import os
#import sys

# 1. 允许重复库运行
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. 【核心修复】强制把 torch 的 lib 目录加入 Windows 搜索路径
#import torch
#torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
#if os.path.exists(torch_lib_path):
#    os.add_dll_directory(torch_lib_path)

# 3. 现在的导入顺序
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os

# 导入咱们刚写的两大核心模块
from dataset import CrossSectionalDataset
from model import CrossSectionalTransformer

# ==========================================
# 1. 固定所有随机种子
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# ==========================================
# 2. 顶级私募的核心科技：Pearson IC 损失函数
# ==========================================
class ICLoss(nn.Module):
    def __init__(self):
        super(ICLoss, self).__init__()

    def forward(self, pred, target):
        # pred 和 target 形状: [1, num_stocks]
        # 计算预测值和真实值的均值
        pred_mean = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)
        
        # 计算标准差 (加上 1e-6 防止除以 0 导致 nan)
        pred_std = pred.std(dim=1, keepdim=True) + 1e-6
        target_std = target.std(dim=1, keepdim=True) + 1e-6
        
        # 计算协方差
        cov = ((pred - pred_mean) * (target - target_mean)).mean(dim=1)
        
        # Pearson 相关系数 (IC)
        ic = cov / (pred_std.squeeze(-1) * target_std.squeeze(-1))
        
        # 因为 PyTorch 的优化器是“最小化”Loss，而我们想要“最大化”IC
        # 所以我们在 IC 前面加个负号
        return -ic.mean()

# ==========================================
# 3. 训练主循环
# ==========================================
def train_v2_model():
    set_seed(42)
    print(" 启动 V2.0 截面量化训练引擎...")

    # 1. 加载 3D 截面数据集
    full_dataset = CrossSectionalDataset(file_path="data/hs300_panel_features.csv")
    total_days = len(full_dataset)
    
    # 2. 严格的时间序列切分 (绝不能随机打乱！)
    # 假设前 80% 的时间作为训练集，后 20% 作为样本外测试集
    train_size = int(total_days * 0.8)
    
    train_dataset = Subset(full_dataset, range(0, train_size))
    test_dataset = Subset(full_dataset, range(train_size, total_days))
    
    # 注意：batch_size 必须为 1，因为每天的股票数量可能不同
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f" 数据切分完毕：训练集 {len(train_dataset)} 天，测试集 {len(test_dataset)} 天。")

    # 3. 初始化模型、优化器和损失函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossSectionalTransformer(feature_dim=5).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = ICLoss() # 使用我们专属的 IC Loss
    
    # 4. 暴力炼丹循环
    epochs = 15
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        valid_days = 0
        
        for batch in train_loader:
            x = batch['X'].to(device)
            y = batch['Y'].to(device)
            
            # 如果某天只有不到 3 只股票，算 IC 没意义，跳过
            if x.shape[1] < 3:
                continue
                
            optimizer.zero_grad()
            predictions = model(x)
            
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            valid_days += 1
            
        avg_loss = total_loss / valid_days if valid_days > 0 else 0
        # Loss 是负的 IC，为了好看，我们把它反转回正的 IC 打印出来
        print(f" Epoch [{epoch+1}/{epochs}] | 训练集平均 IC: {-avg_loss:.4f}")

    print(" V2.0 模型训练完成！")
    
    # 可以在这里加一行保存模型的代码
    # torch.save(model.state_dict(), "models/v2_transformer.pt")

if __name__ == "__main__":
    # 确保有 models 文件夹用于保存权重
    os.makedirs("models", exist_ok=True)
    train_v2_model()