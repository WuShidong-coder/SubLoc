import torch
import torch.nn as nn

from models.legacy.multi_head_attention import MultiHeadAttention


class BioAttention5_5(nn.Module):
    """
    5_1:biGRU+完整自注意力；
    5_2:最后cat拼接；
    5_3:bigru只需获取最后一层维度！！！[:,-1,:]
    5_4:在自注意力中使用mask
    5_5:调整位置编码的顺序
    """
    def __init__(self, embeddings_dim: int = 1024, output_dim: int = 10, dropout=0.25, attention_dropout=0.25, n_heads=8):
        super(BioAttention5_5, self).__init__()

        self.multi_head_attention = MultiHeadAttention(embeddings_dim, attention_dropout, n_heads, skip_last_linear=True)
        self.softmax = nn.Softmax(dim=-1)
        self.bigru = nn.GRU(embeddings_dim, 512, batch_first=True, bidirectional=True)
        self.feedforward = nn.Sequential(
            nn.Linear(embeddings_dim, 2048),
            nn.ReLU(),
            nn.Dropout(attention_dropout),
            nn.Linear(2048, embeddings_dim)
        )
        self.layernorm = nn.LayerNorm(embeddings_dim)
        self.linear = nn.Sequential(
            nn.Linear(embeddings_dim*2, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        x = x.permute(0, 2, 1)  # [batch_size, sequence_length, embeddings_dim]
        x_bi, _ = self.bigru(x)
        x = positional_encoding(x)  # 加上位置编码


        # 自注意力模块 [batch_size, sequence_length, embeddings_dim]
        o, _ = self.multi_head_attention(x, x, x, mask)  # bigru分开，使用mask # 因为前面使用了bigru，这里不再加mask
        o = t = self.layernorm(x + o)
        o = self.feedforward(o)
        o = o + t
        o = self.layernorm(o)

        o = torch.mean(o, dim = 1)  # [batch_size, embeddings_dim]
        # 使用cat拼接attn和bigru
        # x = torch.mean(x, dim = 1)
        x_bi = x_bi[:,-1,:]
        o = torch.cat([o, x_bi], dim=-1)

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