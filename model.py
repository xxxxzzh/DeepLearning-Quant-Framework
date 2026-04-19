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
    
    