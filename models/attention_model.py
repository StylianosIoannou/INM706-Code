import torch
import torch.nn as nn

class AttentionGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        weights = torch.softmax(self.attention(gru_out), dim=1)
        context = torch.sum(weights * gru_out, dim=1)
        out = self.dropout(context)
        out = self.fc(out)
        return self.sigmoid(out).squeeze(1)
