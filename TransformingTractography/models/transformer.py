import numpy as np
import torch
from torch.nn.modules.transformer import (
    Transformer,
    TransformerEncoder,
    TransformerDecoder)

from dwi_ml.models.main_models import MainModelAbstract
from dwi_ml.models.embeddings_on_packed_sequences import keys_to_embeddings
from dwi_ml.models.direction_getter_models import keys_to_direction_getters

from TransformingTractography.models.encoder_decoder import (
    OurTransformerEncoderLayer, OurTransformerDecoderLayer)
from TransformingTractography.models.positional_embeddings import \
    keys_to_positional_embeddings


class TransformingTractographyModel(MainModelAbstract):
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

    Data embedding:
    We could even use the raw data,
    technically. But when adding the positional embedding, the reason it works
    is that the learning of the embedding happens while knowing that some
    positional vector will be added to it. As stated in the blog
    https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    the embedding probably adapts to leave place for the positional embedding.

    All this to say we can't use raw data, and the minimum is to learn to adapt
    with a neural network.
    """

    def __init__(self,
                 experiment_name,
                 # Concerning inputs:
                 neighborhood_type, neighborhood_radius, input_size,
                 # Concerning data embedding:
                 data_embedding_key: str, d_model: int,
                 # Concerning positional embedding:
                 positional_embedding_key: str,
                 # Torch's transformer parameters
                 nhead: int=8,
                 n_layers_e: int = 6, n_layers_d: int = 6,
                 dim_ffnn: int = 2048, dropout_rate: float = 0.1,
                 activation: str = 'relu',
                 # Direction getter
                 direction_getter_key='sphere-classification',
                 # Concerning targets:
                 normalize_directions=True):
        """
        Args
        ----
        d_model: int,
            Data embedding layer must produce outputs of size d_model (i.e.
            number of features at each voxel after embedding).
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
        dropout_rate: float
            Dropout rate. [0.1]
        activation: str
            Choice of activation function in the FFNN. 'relu' or 'gelu'.
            ['relu']
        """
        super().__init__(experiment_name, normalize_directions,
                         neighborhood_type, neighborhood_radius)

        # Params that stay the same throughout all layers
        #  (would it be useful to allow variable values?)
        self.dropout_rate = dropout_rate
        self.dropout_layer = torch.nn.Dropout(dropout_rate)
        self.activation = activation
        self.nhead = nhead

        # Other params:
        self.input_size = input_size
        self.d_model = d_model  # output_size
        self.data_embedding_key = data_embedding_key
        self.positional_embedding_key = positional_embedding_key
        self.n_layers_e = n_layers_e
        self.n_layers_d = n_layers_d

        # -------- Prepare everything

        # Embeddings:
        data_embedding_cls = keys_to_embeddings[self.data_embedding_key]
        self.data_embedding_layer = data_embedding_cls(self.input_size,
                                                  output_size=self.d_model)
        positional_embedding_cls = keys_to_positional_embeddings[
            self.positional_embedding_key]
        self.positional_embedding_layer = positional_embedding_cls()

        # Encoder:
        encoder_layer = OurTransformerEncoderLayer(d_model, nhead, dim_ffnn,
                                                   dropout_rate, activation)
        encoder = TransformerEncoder(encoder_layer, n_layers_e, norm=None)

        # Decoder
        decoder_layer = OurTransformerDecoderLayer(d_model, nhead, dim_ffnn,
                                                   dropout_rate, activation)
        decoder = TransformerDecoder(decoder_layer, n_layers_d, norm=None)

        # Defined in super:
        self.transformer_layer = Transformer(
            d_model, nhead, n_layers_e, n_layers_d, dim_feedforward_2048,
            dropout_rate, activation_F.relu, encore, decoder,
            layer_norm_eps_1em5, batch_first_bool, norm_first_bool, device,
            dtype)


        # Final work on the output
        # Original paper: Linear + Softmax on nb of classes.
        # toDO Original paper: share weights here and in embedding. I don't
        #  think we can to this... To see..
        direction_getter_cls = keys_to_direction_getters[direction_getter_key]
        self.direction_getter_layer = direction_getter_cls(
            self.transformer_layer.output_size, dropout_rate)

    @property
    def params(self):
        p = super().params
        p.update({
            'var1': var1,
        })
        return p

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

        # Embedding (data + positional)
        x = self.data_embedding_layer(x) + self.positional_embedding_layer(x)
        x *= np.sqrt(self.d_emb)  # ToDo. Ils font ça pcq ils sharent the weights of embedding layers and decoder.
        #  Pour l'instant on ne share pas donc peut-être pas nécessaire.
        x = self.dropout(x)

        # Embedding targets?
        # y = self.emb_layer_y(y)

        # Main transformer
        output = self.transformer_layer(
            src=x, tgt=y, src_mask=x_mask, tgt_mask=y_mask,
            memory_mask=encoder_output_mask, src_key_padding_mask=None,
            tgt_key_padding_mask=None, memory_key_padding_mask=None)

        output = self.direction_getter_layer(output)

        return output

    def compute_loss(self, model_outputs, streamlines, device):
        raise NotImplementedError

    def get_tracking_direction_det(self, model_outputs):
        raise NotImplementedError

    def sample_tracking_direction_prob(self, model_outputs):
        raise NotImplementedError
