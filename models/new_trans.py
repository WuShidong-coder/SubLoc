import torch
import torch.nn as nn


class BioAttention3(nn.Module):
    def __init__(self, embeddings_dim=1024, output_dim=10, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(BioAttention3, self).__init__()
        self.bigru = nn.GRU(embeddings_dim, 512, batch_first=True, bidirectional=True)

        self.enconder_layer = nn.TransformerEncoderLayer(embeddings_dim, 4, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.enconder_layer, num_layers=3)
        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        先经过自注意力（transformerencoder）， 再经过bigru。
        Args:
            x: [batch_size, embeddings_dim, sequence_length] = [128,1024,1000] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        # attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        # attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        x = torch.permute(x, (0, 2, 1))
        key_padding_mask =mask
        o = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        o, _ = self.bigru(o)
        # o = torch.permute(o, (0, 2, 1))

        o = torch.mean(x, dim=1)
        # o = self.dropout(o)  # [batch_size, embeddings_dim, sequence_length]


        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)
        #
        # o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim] 求得x`,把对应的累加
        # o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]，求得m，即取最大值
        # o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.softmax(self.output(o))  # [batchsize, output_dim]
