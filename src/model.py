import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, 
                            hidden_dim, 
                            num_layers, 
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0
                            )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, 1, 13, 36) → we need (batch_size, 36, 13)
        x = x.squeeze(1).permute(0, 2, 1)  # (B, 13, 36) → (B, 36, 13)

        # LSTM output
        out, (hn, cn) = self.lstm(x)  # out: (B, 36, H), hn: (num_layers, B, H)

        # Use the last hidden state for classification
        out = self.fc(hn[-1])  # hn[-1] = (B, hidden_size)

        return out