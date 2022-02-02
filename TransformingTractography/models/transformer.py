# -*- coding: utf-8 -*-
import logging

import torch
from dwi_ml.data.processing.streamlines.post_processing import \
    compute_and_normalize_directions
from torch.nn import Dropout
from torch.nn.functional import pad
from torch.nn.modules.transformer import (
    Transformer,
    TransformerEncoder, TransformerDecoder,
    TransformerEncoderLayer, TransformerDecoderLayer)

from dwi_ml.models.main_models import MainModelAbstract
from dwi_ml.models.embeddings_on_tensors import keys_to_embeddings
from dwi_ml.models.direction_getter_models import keys_to_direction_getters

from TransformingTractography.models.positional_encoding import \
    keys_to_positional_encodings

# Pour les masques:
# https://stackoverflow.com/questions/68205894/how-to-prepare-data-for-tpytorchs-3d-attn-mask-argument-in-multiheadattention
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html


def forward_padding(data: torch.tensor, nb_pad):
    return pad(data, (0, 0, 0, nb_pad))


class AbstractTransformerModel(MainModelAbstract):
    """
    Prepares the parts common to our two transformer versions: embeddings,
    direction getter and some parameters for the model.

    Encoder and decoder will be prepared in child classes.

    Child forward methods should look like:
        x, t = self._run_embeddings(x, t)
        outputs = (run main transformer)
        formatted_outputs = self.direction_getter_layer(outputs)

    About data embedding:
    We could even use the raw data, technically. But when adding the positional
    embedding, the reason it works is that the learning of the embedding
    happens while knowing that some positional vector will be added to it.
    As stated in the blog
    https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    the embedding probably adapts to leave place for the positional embedding.

    All this to say we can't use raw data, and the minimum is to learn to adapt
    with a neural network.
    """
    batch_first = True  # If True, then the input and output tensors are
    # provided as (batch, seq, feature). If False, (seq, batch, feature).

    def __init__(self,
                 experiment_name,
                 # Concerning inputs:
                 neighborhood_type: str, neighborhood_radius, nb_features: int,
                 # Concerning embeddings:
                 max_len, positional_encoding_key: str,
                 x_embedding_key: str, t_embedding_key: str,
                 # Torch's transformer parameters
                 d_model: int = 4096, dim_ffnn: int = None, nheads: int = 8,
                 dropout_rate: float = 0.1, activation: str = 'relu',
                 # Direction getter
                 direction_getter_key: str = 'l2-regression',
                 dg_args: dict = None,
                 # Concerning targets:
                 normalize_directions=True):
        """
        Args
        ----
        experiment_name: str
            Name of the experiment.
        neighborhood_type: Union[str, None]
            For usage explanation, see prepare_neighborhood_information.
        neighborhood_radius: Union[int, float, Iterable[float], None]
            For usage explanation, see prepare_neighborhood_information.
        nb_features: int
            This value should be known from the actual data. Number of features
            in the data (last dimension). Size of inputs received during
            training should be nb_features * (nb_neighbors + 1).
        x_embedding_key: str,
            Chosen class for the input embedding (the data embedding part).
            Choices: keys_to_embeddings.keys().
        positional_encoding_key: str,
            Chosen class for the input's positional embedding. Choices:
            keys_to_positional_embeddings.keys().
        t_embedding_key: str,
            Target embedding. See x_embedding_key.
        d_model: int,
            The transformer NEEDS the same output dimension for each layer
            everywhere to allow skip connections. = d_model. Note that
            embeddings should also produce outputs of size d_model.
        dim_ffnn: int
            Size of the feed-forward neural network (FFNN) layer in the encoder
            and decoder layers. The FFNN is composed of two linear layers. This
            is the size of the output of the first one. In the music paper,
            = d_model/2. If None, will use d_model/2.
        nheads: int
            Number of attention heads in each attention or self-attention layer
            [8].
        dropout_rate: float
            Dropout rate [0.1]. Constant in every dropout layer.
        activation: str
            Choice of activation function in the FFNN. 'relu' or 'gelu'.
            ['relu']
        direction_getter_key: str
            Key to the chosen direction getter class. Choices:
            keys_to_direction_getters.keys()
        normalize_directions: bool
            If true, direction vectors are normalized (norm=1). If the step
            size is fixed, it shouldn't make any difference. If streamlines are
            compressed, in theory you should normalize, but you could hope that
            not normalizing could give back to the algorithm a sense of
            distance between points.
        """
        super().__init__(experiment_name, normalize_directions,
                         neighborhood_type, neighborhood_radius)

        self.nb_features = nb_features
        self.max_len = max_len
        self.positional_encoding_key = positional_encoding_key
        self.embedding_key_x = x_embedding_key
        self.embedding_key_t = t_embedding_key
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.nheads = nheads
        self.d_model = d_model
        self.dim_ffnn = dim_ffnn if dim_ffnn is not None else d_model // 2
        self.direction_getter_key = direction_getter_key
        self.dg_args = dg_args

        # ----------- Checks
        if self.embedding_key_x not in keys_to_embeddings.keys():
            raise ValueError("Embedding choice for x data not understood: {}"
                             .format(self.embedding_key_x))
        if self.embedding_key_t not in keys_to_embeddings.keys():
            raise ValueError("Embedding choice for targets not understood: {}"
                             .format(self.embedding_key_t))
        if self.positional_encoding_key not in \
                keys_to_positional_encodings.keys():
            raise ValueError("Positional encoding choice not understood: {}"
                             .format(self.positional_encoding_key))
        if self.direction_getter_key not in keys_to_direction_getters.keys():
            raise ValueError("Direction getter choice not understood: {}"
                             .format(self.positional_encoding_key))

        # ----------- Input size:
        # (neighborhood prepared by super)
        nb_neighbors = len(self.neighborhood_points) if \
            self.neighborhood_points else 0
        self.input_size = nb_features * (nb_neighbors + 1)

        # ----------- Instantiations
        # x embedding
        cls_x = keys_to_embeddings[self.embedding_key_x]
        self.embedding_layer_x = cls_x(self.input_size, d_model)
        # This dropout is only used in the embedding; torch's transformer
        # prepares its own dropout elsewhere.
        self.dropout = Dropout(self.dropout_rate)

        # positional embedding
        cls_p = keys_to_positional_encodings[self.positional_encoding_key]
        self.embedding_layer_position = cls_p(d_model, dropout_rate, max_len)

        # target embedding
        cls_t = keys_to_embeddings[self.embedding_key_t]
        self.embedding_layer_t = cls_t(3, d_model)

        # Last layer
        # Original paper: Linear + Softmax on nb of classes.
        # Us: direction getter
        # Note on parameter initialization.
        # They all use torch.nn.linear, which initializes parameters based
        # on a kaiming uniform, same as uniform(-sqrt(k), sqrt(k)) where k is
        # the nb of features.
        cls_dg = keys_to_direction_getters[self.direction_getter_key]
        self.direction_getter_layer = cls_dg(d_model, **dg_args)

    @property
    def params(self):
        """
        Every parameter necessary to build the different layers again
        from a checkpoint.
        """
        p = super().params
        p.update({
            'nb_features': self.nb_features,
            'x_embedding_key': self.embedding_key_x,
            'positional_embedding_key': self.positional_encoding_key,
            't_embedding_key': self.embedding_key_t,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'nheads': self.nheads,
            'd_model': self.d_model,
            'dim_ffnn': self.dim_ffnn,
            'direction_getter_key': self.direction_getter_key
        })
        return p

    def forward(self, batch_x, batch_s):
        """
        Params
        ------
        batch_x, batch_t: list[Tensor]
            Of length nb_inputs.
        """
        batch_t = compute_and_normalize_directions(batch_s, self.device,
                                                   self.normalize_directions)

        # Padding
        formatted_x, formatted_t = \
            self._pad_and_concatenate_batch(batch_x, batch_t)
        formatted_x.to(self.device)
        formatted_t.to(self.device)

        # Embedding
        embed_x, embed_t = self._embedding_and_positional_encoding_bloc(
            formatted_x, formatted_t)

        # Prepare mask
        mask, padded_masks = self._prepare_masks_batch(batch_x)

        outputs = self._run_main_layer_forward(embed_x, embed_t, mask,
                                               padded_masks)

        # Direction getter
        formatted_outputs = self.direction_getter_layer(outputs)

        return formatted_outputs

    def _prepare_masks_batch(self, batch_x):
        future_mask = Transformer.generate_square_subsequent_mask(self.max_len)

        batch_padded_masks = torch.zeros(len(batch_x), self.max_len)
        for i in range(len(batch_x)):
            x = batch_x[i]
            batch_padded_masks[i, len(x):-1] = float('-inf')

        return future_mask, batch_padded_masks

    def _pad_and_concatenate_batch(self, batch_x, batch_t):
        """
        Padding all streamlines to self.padding_length.
        Concatenating batch on last dim. This is why batch_first = True,
        final result should be of size (batch, seq, feature).
        """
        assert len(batch_x) == len(batch_t), \
            "Error, the batch does not contain the same number of inputs " \
            "and targets..."

        # Padding
        padded_inputs = []
        padded_targets = []
        for s in range(len(batch_x)):
            assert len(batch_x[s]) == len(batch_t[s]), \
                "Error, the input sequence and target sequence do not have " \
                "the same length."
            nb_inputs = len(batch_x[s])
            nb_pad_to_add = self.max_len - nb_inputs
            padded_inputs.append(forward_padding(batch_x[s], nb_pad_to_add))
            padded_targets.append(forward_padding(batch_t[s], nb_pad_to_add))

        # Concatenating
        formatted_x = torch.stack(padded_inputs)
        formatted_t = torch.stack(padded_targets)

        return formatted_x, formatted_t

    def _embedding_and_positional_encoding_bloc(self, x, t):
        """
        First step of the forward pass.

        Params
        ------
        x: Tensor
            Input. Last dim should be of size self.input_size.
        t: Tensor
            Targets. Should be padded sequences of size [max_seq, 3].
        """
        # Embedding (data + positional)
        x = self.embedding_layer_x(x) + self.embedding_layer_position(x)
        x = self.dropout(x)

        # Embedding targets
        t = self.embedding_layer_t(t) + self.embedding_layer_position(t)
        t = self.dropout(t)

        return x, t

    def _run_main_layer_forward(self, embed_x, embed_t, mask, padded_masks):
        raise NotImplementedError

    def compute_loss(self, model_outputs, streamlines, device):
        self.direction_getter_layer.compute_loss(model_outputs, streamlines)

    def get_tracking_direction_det(self, model_outputs):
        self.direction_getter_layer.get_tracking_direction_get(model_outputs)

    def sample_tracking_direction_prob(self, model_outputs):
        self.direction_getter_layer.sample_tracking_direction_prob(
            model_outputs)


class OriginalTransformerModel(AbstractTransformerModel):
    """
    We can use torch.nn.Transformer.
    We will also compare with
    https://github.com/jason9693/MusicTransformer-pytorch.git

                                                 direction getter
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

    """
    layer_norm = 1e-5  # epsilon value for the normalization sub-layers
    norm_first = True  # If True, encoder and decoder layers will perform
    # LayerNorms before other attention and feedforward operations, otherwise
    # after. Torch default + in original paper: False.

    def __init__(self,
                 experiment_name,
                 # Concerning inputs:
                 neighborhood_type, neighborhood_radius, nb_features,
                 # Concerning embedding:
                 max_len, positional_encoding_key: str,
                 x_embedding_key: str, t_embedding_key: str,
                 # Torch's transformer parameters
                 d_model: int = 4096, dim_ffnn: int = None, nheads: int = 8,
                 dropout_rate: float = 0.1, activation: str = 'relu',
                 n_layers_e: int = 6, n_layers_d: int = 6,
                 # Direction getter
                 direction_getter_key: str = 'l2-regression',
                 dg_args: dict = None,
                 # Concerning targets:
                 normalize_directions=True):
        """
        Args
        ----
        n_layers_e: int
            Number of encoding layers in the encoder. [6]
        n_layers_d: int
            Number of encoding layers in the decoder. [6]
        """
        super().__init__(experiment_name, neighborhood_type,
                         neighborhood_radius, nb_features, max_len,
                         positional_encoding_key, x_embedding_key,
                         t_embedding_key, d_model, dim_ffnn, nheads,
                         dropout_rate, activation, direction_getter_key,
                         dg_args, normalize_directions)

        # ----------- Additional params
        self.n_layers_e = n_layers_e
        self.n_layers_d = n_layers_d

        # ----------- Additional instantiations
        logging.info("Instantiating torch transformer, may take a few "
                     "seconds...")
        # Encoder:
        encoder_layer = TransformerEncoderLayer(
            self.d_model, self.nheads, dim_ffnn, self.dropout_rate,
            self.activation, batch_first=True)
        encoder = TransformerEncoder(encoder_layer, n_layers_e, norm=None)

        # Decoder
        decoder_layer = TransformerDecoderLayer(
            self.d_model, self.nheads, dim_ffnn, self.dropout_rate,
            self.activation, batch_first=True)
        decoder = TransformerDecoder(decoder_layer, n_layers_d, norm=None)

        self.transformer_layer = Transformer(
            d_model, nheads, n_layers_e, n_layers_d, dim_ffnn,
            dropout_rate, activation, encoder, decoder,
            self.layer_norm, batch_first=True, norm_first=False)

    @property
    def params(self):
        p = super().params
        p.update({
            'n_layers_e': self.n_layers_e,
            'n_layers_d': self.n_layers_d,
        })
        return p

    def _run_main_layer_forward(self, embed_x, embed_t, mask, padded_mask):
        """Original Main transformer"""

        outputs = self.transformer_layer(
            src=embed_x, tgt=embed_t,
            src_mask=mask, tgt_mask=mask, memory_mask=mask,
            src_key_padding_mask=padded_mask,
            tgt_key_padding_mask=padded_mask,
            memory_key_padding_mask=padded_mask)

        return outputs


class TransformerSourceAndTargetModel(AbstractTransformerModel):
    """
    Decoder only. Concatenate source + target together as input.
    See https://arxiv.org/abs/1905.06596
    + discussion with Hugo.

                                                        direction getter
                                                              |
                                                  -------| take 1/2 |
                                                  |    Norm      x2 |
                                                  |    Skip      x2 |
                                                  |  Dropout     x2 |
                                                  |2-layer FFNN  x2 |
                                                  |        |        |
                                                  |   Norm       x2 |
                                                  |   Skip       x2 |
                                                  |  Dropout     x2 |
                                                  | Masked Att.  x2 |
                                                  -------------------
                                                           |
                                             [ emb_choice_x ; emb_choice_y ]

    """
    def __init__(self,
                 experiment_name,
                 # Concerning inputs:
                 neighborhood_type, neighborhood_radius, nb_features,
                 # Concerning embedding:
                 max_len, positional_encoding_key: str,
                 x_embedding_key: str, t_embedding_key: str,
                 # Torch's transformer parameters
                 d_model: int = 4096, dim_ffnn: int = None, nheads: int = 8,
                 dropout_rate: float = 0.1, activation: str = 'relu',
                 n_layers_d: int = 6,
                 # Direction getter
                 direction_getter_key: str = 'l2-regression',
                 dg_args: dict = None,
                 # Concerning targets:
                 normalize_directions=True):
        """
        Args
        ----
        n_layers_d: int
            Number of encoding layers in the decoder. [6]
        """
        super().__init__(experiment_name, neighborhood_type,
                         neighborhood_radius, nb_features, max_len,
                         positional_encoding_key, x_embedding_key,
                         t_embedding_key, d_model, dim_ffnn, nheads,
                         dropout_rate, activation, direction_getter_key,
                         dg_args, normalize_directions)

        # ----------- Additional params
        self.n_layers_d = n_layers_d

        # ----------- Additional instantiations
        # We say "decoder only" from the logical point of view, but code-wise
        # it is actually "encoder only". A decoder would need output from the
        # encoder.
        double_layer = TransformerEncoderLayer(
            self.d_model * 2, self.nheads, dim_ffnn, self.dropout_rate,
            self.activation)
        self.main_layer = TransformerEncoder(double_layer, n_layers_d,
                                             norm=None)

        self._instantiate_direction_getter(self.transformer_layer.output_size,
                                           dg_args)

    def _run_main_layer_forward(self, embed_x, embed_t, mask, padded_mask):

        # Concatenating x and t
        inputs = torch.cat((embed_x, embed_t), dim=-1)
        self.logger.debug("Concatenated [src | tgt] shape: {}"
                          .format(inputs.shape))

        # Doubling the masks
        double_mask = torch.cat((mask, mask), dim=-1)
        double_padded_mask = torch.cat((padded_mask, padded_mask), dim=-1)
        self.logger.debug("Concatenated mask shape: {}".format(inputs.shape))

        # Main transformer
        outputs = self.main_layer(src=inputs, src_mask=double_mask,
                                  src_key_padding_mask=double_padded_mask)

        # Take the second half of model outputs to direction getter
        # (the last skip-connection makes more sense this way)
        return outputs[-self.d_model:-1]
