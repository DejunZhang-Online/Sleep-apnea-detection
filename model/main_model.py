from seg_level_encoder import Seg_Encoder
from seq_level_encoder import Seq_Encoder
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes=2, seq_length=20):
        super(Model, self).__init__()
        hidden_dim = 128

        self.seg_encoder = Seg_Encoder()
        self.seq_encoder = Seq_Encoder(seq_length=seq_length,
                                      num_layers=1,
                                      num_heads=8,
                                      hidden_dim=hidden_dim,
                                      mlp_dim=hidden_dim,
                                      dropout=0.1,
                                      attention_dropout=0.1,
                                      )

        self.main_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

        self.trans_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        batch_size, num_epochs, num_channels, num_samples = x.shape
        x = x.view(batch_size * num_epochs, num_channels, -1)
        x1, x2 = x[:, :1, :], x[:, -1:, :]
        x = self.seg_encoder(x1, x2)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(batch_size, num_epochs, -1)
        x = self.seq_encoder(x)
        y = self.main_head(x)
        y_t = self.trans_head(x)

        return y, y_t


if __name__ == '__main__':
    model = Model(seq_length=60)
    data = torch.randn(2, 60, 2, 180)
    y, y_t = model(data)
    print(y.shape, y_t.shape)

