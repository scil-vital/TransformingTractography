
Torch has already implemented encoder and decoder layers exactly as in the
original paper.

Encoder:

    Paper figure 1: Note. See section 3.1 for the skip connection +
    normalization. See section 3.3 for the FFNN. See section 5.4 for the
    dropout layers.

              ENCODER
           --------------
           |    Norm2   |
           |    Skip    |
           |   Dropout  |
           |    FFNN**  |                       ** FFNN:
           |     |      |                     -------------
           |    Norm1   |                     |   Linear2 |
           |    Skip    |                     |   Dropout |
           |  Dropout   |                     |   Relu    |
           | Attention  |                     |   Linear1 |
           --------------                     -------------
                 |
            emb_choice_x

    Torch's encoder:
            - First 4 lines = _sa_bloc (self-attention). Possibility to
            put the norm first (default false). Uses
            self.self_attn = MultiheadAttention. So we will need to change
            that here if we want to change the type of attention (ex:
            shape-attention).
            - Lines 5-8: _ff_bloc (feed-forward).

Decoder:

    Paper figure 1: Note. See section 3.1 for the skip connection +
    normalization. See section 3.3 for the FFNN. See section 5.4 for the
    dropout layers.
                              DECODER
                          --------------
                          |    Norm3   |
                          |    Skip    |
                          |  Dropout   |
                          |2-layer FFNN|
                          |      |     |
                          |    Norm2   |
                          |    Skip    |
        encoder_output    |  Dropout   |
        ------------->    | Attention  |
                          |      |     |
                          |   Norm1    |
                          |   Skip     |
                          |  Dropout   |
                          | Masked Att.|
                          --------------

    Torch:
        - First 4 line = _sa_block = self-attention
        - Lines 4-8 = _mha_block = multi-head attention
        - Lines 9-12 = _ff_block.
