import torch
import torch.nn as nn


class BioAttention2_4(nn.Module):
    def __init__(self, embeddings_dim=1024, output_dim=10, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(BioAttention2_4, self).__init__()
        self.bigru = nn.GRU(embeddings_dim, 512, 3, batch_first=True, bidirectional=True)
        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(3 * embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        先经过bigru，再经过LA，最后输出
        Args:
            x: [batch_size, embeddings_dim, sequence_length] = [128,1024,1000] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """


        x = torch.permute(x, (0, 2, 1))  # [batch_size, length, embd_dim]
        x_bi, _ = self.bigru(x)
        # o = torch.permute(o, (0, 2, 1))  # [batch_size, embeddings_dim, sequence_length]

        x = torch.permute(x, (0, 2, 1))  # [batch_size, embeddings_dim, sequence_length]
        # o = x + o
        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout(o)  # [batch_size, embeddings_dim, sequence_length]
        attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)
        x_bi = x_bi[:,-1,:]
        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim] 求得x`,把对应的累加
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]，求得m，即取最大值
        o = torch.cat([o1, o2, x_bi], dim=-1)  # [batchsize, 3*embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.softmax(self.output(o))  # [batchsize, output_dim]
