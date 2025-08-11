import torch
import torch.nn as nn
from models.legacy.multi_head_attention import MultiHeadAttention


class SubLoc02(nn.Module):
    def __init__(self, embeddings_dim: int = 1024, output_dim: int = 12 , dropout=0.25, attention_dropout=0.25, n_heads=8):
        super(SubLoc02, self).__init__()

        self.multi_head_attention = MultiHeadAttention(embeddings_dim, attention_dropout, n_heads, skip_last_linear=True)
        self.bigru = nn.GRU(embeddings_dim, 512, batch_first=True, num_layers=3, bidirectional=True)
        self.gru = nn.GRU(embeddings_dim, 1024, batch_first=True, num_layers=1)
        self.rnn = nn.RNN(embeddings_dim, 1024, batch_first=True, num_layers=1)
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Sequential(
            nn.Linear(embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor, mask, sequence_lengths, frequencies) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        # query = x.mean(dim=-1)  # [batch_size, embeddings_dim]
        x = x.permute(0, 2, 1)  # [batch_size, sequence_length, embeddings_dim]
        # x_bi, _ = self.bigru(x)  # 双向gru使用
        # x_gru, _ = self.gru(x)  # gru使用
        x_rnn, _ = self.rnn(x)
        query = x_rnn[:,-1,:]
        o, _ = self.multi_head_attention(query, x, x, mask)  # [batch_size, 1, embeddings_dim]
        o = o.squeeze()  # [batch_size, embeddings_dim]

        o = self.linear(o)
        return self.softmax(self.output(o))

