import torch
from torch.nn.modules.transformer import (
    Transformer,
    TransformerEncoder,
    TransformerDecoder)
from torch.nn.modules.linear import Linear
from torch.nn import Softmax

from TransformingTractography.models.encoder_decoder import (
    OurTransformerEncoderLayer, OurTransformerDecoderLayer)


class OurTransformer(Transformer):
    """
    We can use torch.nn.Transformer.
    We will also compare with
    https://github.com/jason9693/MusicTransformer-pytorch.git

    Note. We are creating here a transformer with length(X) = length(Y).
    In torch: length(X) = S. length(Y) = T. Here, both are named = L.

                                                    out probs
                                                        |
                                                      linear
                                                        |
                                                     DECODER
                                                  --------------
                                                  |    Norm    |
                                                  |    Skip    |
                                                  |  Dropout   |
                                                  |2-layer FFNN|
                  ENCODER                         |      |     |
               --------------                     |    Norm    |
               |    Norm    | ---------           |    Skip    |
               |    Skip    |         |           |  Dropout   |
               |  Dropout   |         --------->  | Attention  |
               |2-layer FFNN|                     |      |     |
               |     |      |                     |   Norm     |
               |    Norm    |                     |   Skip     |
               |    Skip    |                     |  Dropout   |
               |  Dropout   |                     | Masked Att.|
               | Attention  |                     --------------
               --------------                            |
                     |                             emb_choice_y
                emb_choice_x

    Also, TransformingTractography NEED the same output dimension for each layers
    everywhere to allow skip connections. = d_model. Note that emb_choice_x and
    emb_choice_y must also produce outputs of size d_model. They should have a
    self.d_model feature so that we can verify this.
    """

    def __init__(self,
                 # Concerning embedding:
                 emb_layer_x: torch.nn.Module, emb_layer_y: torch.nn.Module,
                 # Concerning the final output:
                 nb_classes: int,
                 # Torch's parameters
                 d_model: int, nhead: int=8,
                 n_layers_e: int = 6, n_layers_d: int = 6,
                 dim_ffnn: int = 2048, dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Args
        ----
        emb_layer_x: torch.nn.Module
            Choice for the positional embedding technique. For example, this
            could be a sinusoidal embedding such as defined in
            VITALabAI.models.generative.TransformingTractography.utils.embedding, a CNN or
            a fully connected neural network. It should be pre-instantiated with
            the parameter choices.  Must produce outputs of size d_model. Should
            have a self.d_model feature so that we can verify this.
        emb_layer_y:  torch.nn.Module
            Idem
        nb_classes: int
            Nb of output choices
        d_model: int
            Size of input modified to add position embedding.
        nhead: int
            Number of heads [8]
        n_layers_e: int
            Number of encoding layers in the encoder. [6]
        n_layers_d: int
            Number of encoding layers in the decoder. [6]
        dim_ffnn: int
            Size of the feed-forward neural network (FFNN) layer in the encoder
            and decoder layers. The FFNN is composed of two linear layers. This
            is the size of the output of the first one. In the music paper,
            = d_model/2. [2048]
        dropout: float
            Dropout rate. [0.1]
        activation: str
            Choice of activation function in the FFNN. 'relu' or 'gelu'.
            ['relu']
        """
        # Verify that the embedding layers produce outputs of size d_model:
        if (emb_layer_x != d_model) or (emb_layer_y != d_model):
            raise ValueError("The embedding layers must produce outputs of dim"
                             "d_model!")

        # Defining our encoder
        encoder_layer = OurTransformerEncoderLayer(d_model, nhead, dim_ffnn,
                                                   dropout, activation)
        encoder_norm = None  # Originally in torch: LayerNorm(d_model), but the
                             # encoder layers already have a norm! Redundant
        custom_encoder = TransformerEncoder(encoder_layer, n_layers_e,
                                            encoder_norm)

        # Defining our decoder
        decoder_layer = OurTransformerDecoderLayer(d_model, nhead, dim_ffnn,
                                                   dropout, activation)
        decoder_norm = None  # Originally in torch: LayerNorm(d_model), but the
                             # decoder layers already have a norm! Redundant
        custom_decoder = TransformerDecoder(decoder_layer, n_layers_d,
                                            decoder_norm)

        super().__init__(d_model=d_model, nhead=nhead,
                         num_encoder_layers=n_layers_e,
                         num_decoder_layers=n_layers_d,
                         dim_feedforward=dim_ffnn, dropout=dropout,
                         activation=activation, custom_encoder=custom_encoder,
                         custom_decoder=custom_decoder)
        # Defined in super:
        # self.encoder = custom_encoder
        # self.decoder = custom_decoder
        # self.d_model = d_model
        # self.nhead = nheads

        # Used in super's init but value not remembered:
        self.dropout_rate = dropout
        self.n_layers_e = n_layers_e
        self.n_layers_d = n_layers_d

        # Embedding
        self.emb_layer_x = emb_layer_x
        self.emb_layer_y = emb_layer_y

        # Final work on the output
        self.final_linear = Linear(d_model, nb_classes)                                                                # toDO Original paper: share weights here and in embedding. I don't
                                                                                                                       #  think we can to this... To see..
        self.final_softmax = Softmax()                                                                                 # Possible argument: dimension along which we want to work.

    def forward(self, x, y, x_mask=None, y_mask=None, encoder_output_mask=None,
                **kwargs):
        """
        Masks should be filled with float('-inf') for the masked positions
        and float(0.0) else.

        Runs embedding + super's forward + managing the output

        1) Embedding: calling the embedding's forward by using it.
            x = embedding_x(x)
            y = embedding_y(y)

        2) Super's forward. What it does:
            encoder_output = encoder(x)       # uses masks
            output = decoder(y, encoder_output)  # uses masks

        3) Managing the output:
            output = linear(output)
        """

        # kwargs should be empty. Just to fit torch's signature. But we don't
        # care about the key_padding args for now. Setting them to none.
        if kwargs is not None:
            raise ValueError("Excedent args not understood.")

        # Embedding
        x = self.emb_layer_x(x)
        y = self.emb_layer_y(y)

        output = super().forward(src=x, tgt=y, src_mask=x_mask, tgt_mask=y_mask,
                                 memory_mask=encoder_output_mask,
                                 src_key_padding_mask=None,
                                 tgt_key_padding_mask=None,
                                 memory_key_padding_mask=None)

        output = self.final_linear(output)
        output_probs = self.final_softmax(output)

        return output_probs

    def generate(self):
        raise NotImplementedError
