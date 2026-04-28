import torch
import torch.nn as nn

class CrossSectionalTransformer(nn.Module):
    def __init__(self, feature_dim=5, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(CrossSectionalTransformer, self).__init__()
        
        # 1. 神经元升维：把 5 维特征映射到高维脑区
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        # 2. 截面注意力机制 (Cross-Sectional Attention)
        # 这里是整个 V2.0 的灵魂：让同时期的股票互相“观察”
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True  # 极其关键！声明输入形状为 [Batch, 股票数, 特征]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 最终打分头：给每一只股票输出一个相对强弱分数 (Alpha Score)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x 形状: [batch_size=1, num_stocks, feature_dim]
        """
        # 第一步：特征升维
        out = self.input_projection(x)  # -> [1, num_stocks, 64]
        
        # 第二步：多头注意力特征交叉 (股票间的心电感应)
        out = self.transformer_encoder(out)  # -> [1, num_stocks, 64]
        
        # 第三步：降维输出单一得分
        out = self.output_layer(out)  # -> [1, num_stocks, 1]
        
        # 压平最后一个维度，使得输出形状完美对应目标 Y: [1, num_stocks]
        return out.squeeze(-1)

# =========================================
# 本地防呆测试模块
# =========================================
if __name__ == "__main__":
    # 模拟你刚才输出的 2020-02-14 的那一天的数据形状
    batch_size = 1
    num_stocks = 4   # 停牌了1只，剩4只
    feature_dim = 5  # 5大核心指标
    
    # 随机生成一个假数据假装是 DataLoader 送进来的
    dummy_x = torch.randn(batch_size, num_stocks, feature_dim)
    
    print(f" 模型输入形状 (X): {dummy_x.shape}")
    
    # 初始化大脑
    model = CrossSectionalTransformer(feature_dim=5)
    
    # 前向传播
    predictions = model(dummy_x)
    
    print(f" 模型输出形状 (Predictions): {predictions.shape}")
    print("-------------------------------------------------")
    print("模型输出的截面打分示例:", predictions.detach().numpy())
    print("大功告成！输出形状完美对齐目标 Y！")
    