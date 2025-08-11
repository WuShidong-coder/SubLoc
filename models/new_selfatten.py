import torch
import torch.nn as nn

from models.legacy.multi_head_attention import MultiHeadAttention


class BioAttention4_3(nn.Module):
    """
    4_2:加入了 残差连接；
    4_3:加入layernorm（层数少用post，层数多用pre）；
    """
    def __init__(self, embeddings_dim: int = 1024, output_dim: int = 10, dropout=0.25, attention_dropout=0.1, n_heads=8):
        super(BioAttention4_3, self).__init__()

        self.multi_head_attention = MultiHeadAttention(embeddings_dim, attention_dropout, n_heads, skip_last_linear=True)
        self.softmax = nn.Softmax(dim=-1)

        self.feedforward = nn.Sequential(
            nn.Linear(embeddings_dim, 2048),
            nn.ReLU(),
            nn.Dropout(attention_dropout),
            nn.Linear(2048, embeddings_dim)
        )
        self.layernorm = nn.LayerNorm(embeddings_dim)
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
        x = x.permute(0, 2, 1)  # [batch_size, sequence_length, embeddings_dim]
        x = positional_encoding(x)
        o, _ = self.multi_head_attention(x, x, x, mask)  # [batch_size, len, embeddings_dim]
        o = t = self.layernorm(x + o)
        o = self.feedforward(o)
        o = o + t
        o = self.layernorm(o)
        o = torch.mean(o, dim=1)  # [batch_size, embeddings_dim]

        o = self.linear(o)
        o = self.output(o)
        return self.softmax(o)

def positional_encoding(X, num_features=1024, dropout_p=0.1, max_len=1000):
    r'''
        给输入加入位置编码
    参数：
        - num_features: 输入进来的维度
        - dropout_p: dropout的概率，当其为非零时执行dropout
        - max_len: 句子的最大长度

    形状：
        - 输入： [batch_size, seq_length, num_features]
        - 输出： [batch_size, seq_length, num_features]
    '''

    dropout = nn.Dropout(dropout_p)
    P = torch.zeros((1,max_len,num_features))
    X_ = torch.arange(max_len,dtype=torch.float32).reshape(-1,1) / torch.pow(10000,torch.arange(0,num_features,2,dtype=torch.float32) /num_features)
    P[:,:,0::2] = torch.sin(X_)
    P[:,:,1::2] = torch.cos(X_)
    X = X + P[:,:X.shape[1],:].to(X.device)
    return dropout(X)