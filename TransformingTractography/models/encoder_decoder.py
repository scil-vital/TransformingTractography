from torch.nn.modules.transformer import (
    TransformerEncoderLayer, TransformerDecoderLayer)

"""
 The encoder and decoder simply iterate over the encoding/decoding layers.
 
 Here, we offer the possibility to re-implement these layers. So far, we
 only call torch's (super's) functions. But we checked that eveything was
 as in the figure of the architecture.
"""

# No need to reimplement the total TransformerEncoder. Passes through each
# encoderLayer (+ layerNorm at the end) so we only need to change the layers.


class OurTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model: int, nhead: int = 8, dim_ffnn: int = 2048,
                 dropout: float = 0.1, activation: str = "relu"):
        """
        Paper figure 1: Note. See section 3.1 for the skip connection +
        normalization. See section 3.3 for the FFNN. See section 5.4 for the
        dropout layers.

                  ENCODER
               --------------
               |    Norm    |
               |    Skip    |
               |   Dropout  |
               |    FFNN**  |                       ** FFNN:
               |     |      |                     -------------
               |    Norm    |                     |   Linear1 |
               |    Skip    |                     |   Dropout |
               |  Dropout   |                     |   Relu    |
               | Attention  |                     |   Linear2 |
               --------------                     -------------
                     |
                emb_choice_x
        """
        super().__init__(d_model, nhead, dim_ffnn, dropout, activation)

    def forward(self, x, x_mask=None, x_key_padding_mask=None):
        """
        x_init = x
        x = self_attention(x)       #(uses masks)                                                                                    # toDo . À voir et comprendre. MultiHeadAttention
        x = x_init + dropout1(x)       # dropout, then skip
        x = norm(x)

        # Feed-forward:
        x_init = x
        x = linear1(x)
        x = activation(x)  # i.e. Relu
        x = dropout(x)              # !! Not clear that it's in the paper!
        x = linear2(x)

        x = x_init + dropout2(x)  # dropout, then skip
        x = norm2(x)
        """
        super().forward(src=x, src_mask=x_mask,
                        src_key_padding_mask=x_key_padding_mask)


class OurTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        """
        Paper figure 1: Note. See section 3.1 for the skip connection +
        normalization. See section 3.3 for the FFNN. See section 5.4 for the
        dropout layers.
                                  DECODER
                              --------------
                              |    Norm    |
                              |    Skip    |
                              |  Dropout   |
                              |2-layer FFNN|
                              |      |     |
                              |    Norm    |
                              |    Skip    |
            encoder_output    |  Dropout   |
            ------------->    | Attention  |
                              |      |     |
                              |   Norm     |
                              |   Skip     |
                              |  Dropout   |
                              | Masked Att.|
                              --------------
        """
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

    def forward(self, y, encoder_output, y_mask=None, encoder_output_mask=None,
                y_key_padding_mask=None, encoder_output_key_padding_mask=None):
        """
        y_init = y
        y = self_attention(y)           # uses masks                                                                                # toDo . À voir et comprendre. MultiHeadAttention
        y = y_init + dropout1(y)       # dropout, then skip
        y = norm(y)

        y_init = y
        y = attention(y, encoder_output)         # uses masks                                                                                 # toDo . À voir et comprendre. MultiHeadAttention
        y = y_init + dropout1(y)       # dropout, then skip
        y = norm2(y)

        # Feed-forward:
        y_init = y
        y = linear1(y)
        y = activation(yx)  # i.e. Relu
        y = dropout(y)              # !! Not clear that it's in the paper!
        y = linear2(y)

        y = y_init + dropout3(y)  # dropout, then skip
        y = norm3(y)
        """
        super().forward(y, encoder_output, y_mask, encoder_output_mask,
                        y_key_padding_mask, encoder_output_key_padding_mask)
