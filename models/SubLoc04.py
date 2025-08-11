import torch
import torch.nn as nn

from models.legacy.multi_head_attention import MultiHeadAttention


class SubLoc04(nn.Module):
    def __init__(self, embeddings_dim: int = 1024, output_dim: int = 12 , dropout=0.25, attention_dropout=0.25, n_heads=8):
        super(SubLoc04, self).__init__()

        self.multi_head_attention = MultiHeadAttention(embeddings_dim, attention_dropout, n_heads, skip_last_linear=True)
        self.multi_head_attention1 = MultiHeadAttention(embeddings_dim, attention_dropout, n_heads)
        self.multi_head_attention2 = MultiHeadAttention(embeddings_dim, attention_dropout, n_heads,
                                                        skip_last_linear=True)
        self.bigru = nn.GRU(embeddings_dim, 512, batch_first=True, bidirectional=True)
        #self.bigru = nn.GRU(embeddings_dim, 512, num_layers=2, batch_first=True, bidirectional=True)
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Sequential(
            nn.Linear(2*embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor, mask, sequence_lengths, frequencies,apply_mutation=True) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        # # 使用两层多头注意力
        # x = x.permute(0, 2, 1)  # [batch_size, sequence_length, embeddings_dim]
        # x, _ = self.bigru(x)
        # o, _ = self.multi_head_attention1(x, x, x, mask)
        # query = torch.sum(o * mask[..., None], dim=-2) / mask[..., None].sum(dim=-2)  # [batch_size, embeddings_dim]
        # o, _ = self.multi_head_attention2(query, o, o, mask)  # [batch_size, 1, embeddings_dim]

        # 使用一层多头注意力
        if apply_mutation:
        	x = random_mutation(x)
        query = x.mean(dim=-1)  # [batch_size, embeddings_dim]
        x = x.permute(0, 2, 1)  # [batch_size, sequence_length, embeddings_dim]
        x_bi, _ = self.bigru(x)
        o, _ = self.multi_head_attention(query, x, x, mask)  # [batch_size, 1, embeddings_dim]
        # print('before:', o.shape)
        # o = o[:, -1, :] # 取最后一个时间步
        # print('after:', o.shape)
        o = o.squeeze()  # [batch_size, embeddings_dim]
        x_bi = x_bi[:,-1,:]
        o = torch.cat([o, x_bi], dim=-1)
        o = self.linear(o)
        return self.softmax(self.output(o))

