# -*- coding: utf-8 -*-

"""
Positional embeddings:
    - SinusoidalPosEmbedding
    - RelationalSinusoidalPosEmbedding
"""
import math
import torch


class AbstractPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout_rate, max_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        self.dropout = torch.nn.Dropout(p=dropout_rate)


class SinusoidalPositionalEncoding(AbstractPositionalEncoding):
    """
    Creates the vector for positional embedding that can be added to the data
    embedding as first layer of a transformer. Such as described in the paper
    Attention is all you need.

    Version1, copied from
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model: int, dropout_rate, max_len: int = 2048):
        """
        Note. Alternate computation:

        pos_emb = np.zeros((self.max_seq, self.d_model))
        for index in range(0, self.d_model, 2):
            pos_emb[:, index] = np.array(
                [math.sin(pos / 10000 ** (index / self.d_model))
                 for pos in range(self.max_seq)])
            pos_emb[:, index + 1] = np.array(
                [math.cos(pos / 10000 ** (index / self.d_model))
                 for pos in range(self.max_seq)])
        """
        super().__init__(d_model, dropout_rate, max_len)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))
        pos_emb = torch.zeros(max_len, 1, d_model)
        pos_emb[:, 0, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 0, 1::2] = torch.cos(position * div_term)

        # pos_emb is a parameter, but not learned. We don't want the optimizer
        # to update this. We could do self.pos_emb= pos_emb.
        # However, it is big enough that we would like it to be moved to GPU or
        # CPU whenever the module is. That is the use of a "buffer".
        self.register_buffer('pos_emb', pos_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return self.pos_emb[:x.size(0)]


class RelationalSinusoidalPosEncoding(AbstractPositionalEncoding):
    """
    Creates the vector for positional embedding that can be added to the data
    embedding as first layer of a transformer. Such as described in the music
    paper to replace the sinusoidal version.
    """
    def __init__(self):
        super().__init__()
        raise NotImplementedError


keys_to_positional_encodings = {
    'sinusoidal': SinusoidalPositionalEncoding,
    'relational': RelationalSinusoidalPosEncoding,
}
