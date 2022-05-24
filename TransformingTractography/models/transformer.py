# -*- coding: utf-8 -*-
import logging
from typing import Union, List

import torch
from torch.nn import Dropout
from torch.nn.functional import pad
from torch.nn.modules.transformer import (
    Transformer,
    TransformerEncoder, TransformerDecoder,
    TransformerEncoderLayer, TransformerDecoderLayer)

from dwi_ml.models.main_models import MainModelWithPD
from dwi_ml.models.embeddings_on_tensors import keys_to_embeddings
from dwi_ml.models.direction_getter_models import keys_to_direction_getters
from torch.nn.utils.rnn import pack_sequence, PackedSequence, unpack_sequence

from TransformingTractography.models.positional_encoding import \
    keys_to_positional_encodings

# Pour les masques:
# https://stackoverflow.com/questions/68205894/how-to-prepare-data-for-tpytorchs-3d-attn-mask-argument-in-multiheadattention
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html

logger = logging.getLogger('model_logger')  # Same logger as Super.


def forward_padding(data: torch.tensor, nb_pad):
    # data: tensor of size
    return pad(data, (0, 0, 0, nb_pad))


class AbstractTransformerModel(MainModelWithPD):
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
                 experiment_name: str, nb_features: int,
                 # PREVIOUS DIRS
                 nb_previous_dirs: int = 0,
                 prev_dirs_embedding_size: int = None,
                 prev_dirs_embedding_key: str = None,
                 # INPUTS
                 max_len: int = 3500,
                 positional_encoding_key: str = 'sinusoidal',
                 x_embedding_key: str = 'nn_embedding',
                 t_embedding_key: str = 'nn_embedding',
                 # TRANSFORMER
                 d_model: int = 4096, dim_ffnn: int = None, nheads: int = 8,
                 dropout_rate: float = 0.1, activation: str = 'relu',
                 # DIRECTION GETTER
                 dg_key: str = 'cosine-regression', dg_args: dict = None,
                 # Other
                 neighborhood_type: str = None,
                 neighborhood_radius: Union[int, float, List[float]] = None,
                 normalize_directions=True,
                 log_level=logging.root.level):
        """
        Args
        ----
        experiment_name: str
            Name of the experiment.
        nb_features: int
            This value should be known from the actual data. Number of features
            in the data (last dimension).
        nb_previous_dirs: int
            Number of previous direction to concatenate to each input.
            Default: 0.
        prev_dirs_embedding_size: int
            Dimension of the final vector representing the previous directions
            (no matter the number of previous directions used).
            Default: nb_previous_dirs * 3.
        prev_dirs_embedding_key: str,
            Key to an embedding class (one of
            dwi_ml.models.embeddings_on_tensors.keys_to_embeddings).
            Default: None (no previous directions added).
        positional_encoding_key: str,
            Chosen class for the input's positional embedding. Choices:
            keys_to_positional_embeddings.keys(). Default: 'sinusoidal'.
        x_embedding_key: str,
            Chosen class for the input embedding (the data embedding part).
            Choices: keys_to_embeddings.keys().
            Default: 'no_embedding'.
        t_embedding_key: str,
            Target embedding, with the same choices as above.
            Default: 'no_embedding'.
        d_model: int,
            The transformer REQUIRES the same output dimension for each layer
            everywhere to allow skip connections. = d_model. Note that
            embeddings should also produce outputs of size d_model.
            Value must be divisible by num_heads.
            Default: 4096.
        dim_ffnn: int
            Size of the feed-forward neural network (FFNN) layer in the encoder
            and decoder layers. The FFNN is composed of two linear layers. This
            is the size of the output of the first one. In the music paper,
            = d_model/2. Default: d_model/2.
        nheads: int
            Number of attention heads in each attention or self-attention
            layer. Default: 8.
        dropout_rate: float
            Dropout rate. Constant in every dropout layer. Default: 0.1.
        activation: str
            Choice of activation function in the FFNN. 'relu' or 'gelu'.
            Default: 'relu'.
        dg_key: str
            Key to the chosen direction getter class. Choices:
            keys_to_direction_getters.keys(). Default: 'cosine-regression'.
        neighborhood_type: str
            The type of neighborhood to add. One of 'axes', 'grid' or None. If
            None, don't add any. See
            dwi_ml.data.processing.space.Neighborhood for more information.
        neighborhood_radius : Union[int, float, Iterable[float]]
            Add neighborhood points at the given distance (in voxels) in each
            direction (nb_neighborhood_axes). (Can be none)
                - For a grid neighborhood: type must be int.
                - For an axes neighborhood: type must be float. If it is an
                iterable of floats, we will use a multi-radius neighborhood.
        normalize_directions: bool
            If true, direction vectors are normalized (norm=1). If the step
            size is fixed, it shouldn't make any difference. If streamlines are
            compressed, in theory you should normalize, but you could hope that
            not normalizing could give back to the algorithm a sense of
            distance between points.
        """
        super().__init__(experiment_name, nb_previous_dirs,
                         prev_dirs_embedding_key, prev_dirs_embedding_size,
                         normalize_directions, neighborhood_type,
                         neighborhood_radius, log_level)

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
        self.dg_key = dg_key
        self.dg_args = dg_args or {}

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
        if self.dg_key not in keys_to_direction_getters.keys():
            raise ValueError("Direction getter choice not understood: {}"
                             .format(self.positional_encoding_key))

        # ----------- Input size:
        # (neighborhood prepared by super)
        nb_neighbors = len(self.neighborhood_points) if \
            self.neighborhood_points else 0
        self.input_size = nb_features * (nb_neighbors + 1)

        # ----------- Instantiations
        # 1. Previous dirs embedding: prepared by super.

        # 2. x embedding
        cls_x = keys_to_embeddings[self.embedding_key_x]
        # output will be concatenated with prev_dir embedding and total must
        # be d_model.
        if self.nb_previous_dirs > 0:
            embed_size = d_model - self.prev_dirs_embedding_size
        else:
            embed_size = d_model
        self.embedding_layer_x = cls_x(self.input_size, embed_size)
        # This dropout is only used in the embedding; torch's transformer
        # prepares its own dropout elsewhere.
        self.dropout = Dropout(self.dropout_rate)

        # 3. positional embedding
        cls_p = keys_to_positional_encodings[self.positional_encoding_key]
        self.embedding_layer_position = cls_p(d_model, dropout_rate, max_len)

        # 4. target embedding
        cls_t = keys_to_embeddings[self.embedding_key_t]
        self.embedding_layer_t = cls_t(3, d_model)

        # 5. Transformer: See child classes

        # 6. Direction getter
        # Original paper: last layer = Linear + Softmax on nb of classes.
        # Note on parameter initialization.
        # They all use torch.nn.linear, which initializes parameters based
        # on a kaiming uniform, same as uniform(-sqrt(k), sqrt(k)) where k is
        # the nb of features.
        cls_dg = keys_to_direction_getters[self.dg_key]
        self.direction_getter_layer = cls_dg(d_model, **self.dg_args)

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
            'direction_getter_key': self.dg_key
        })
        return p

    def forward(self, batch_x, batch_streamlines, version=1):
        """
        Params
        ------
        batch_x, batch_t: list[Tensor]
            Of length nb_inputs.
        """
        dirs = self.format_directions(batch_streamlines)

        if version == 1:
            # We could do like in Learn2track and run the embedding on the
            # PackedSequence's data. It would run faster (no need to run a
            # bunch of zeros + concatenated with previous dir's embedding more
            # easily.
            embed_x, embed_t = self.run_embedding_version1(batch_x, dirs)
        else:
            # Or, we do like in original paper. Run on padded data. As
            # explained in doc above, probably necessary to make the model
            # learn embedding on whole padded sequence so that it can adapt for
            # the positional encoding.
            embed_x, embed_t = self.run_embeding_version2(batch_x, dirs)
        embed_x = self.dropout(embed_x)
        embed_t = self.dropout(embed_t)

        logger.debug("*** 5. Transformer....")

        # Prepare mask
        mask, padded_masks = self._prepare_masks_batch(batch_x)

        outputs = self._run_main_layer_forward(embed_x, embed_t, mask,
                                               padded_masks)

        # Direction getter
        formatted_outputs = self.direction_getter_layer(outputs)

        return formatted_outputs

    def _pad_and_stack_batch(self, batch_x, batch_t):
        padded_inputs = []
        padded_targets = []
        for i in range(len(batch_x)):
            assert len(batch_x[i]) == len(batch_t[i]), \
                "Error, the input sequence and target sequence do not have " \
                "the same length."
            nb_inputs = len(batch_x[i])
            nb_pad_to_add = self.max_len - nb_inputs
            padded_inputs.append(forward_padding(batch_x[i], nb_pad_to_add))
            logger.debug("Final data shape for this streamline: {}"
                         .format(padded_inputs[-1].shape))
            padded_targets.append(forward_padding(batch_t[i], nb_pad_to_add))

        formatted_x = torch.stack(padded_inputs).to(self.device)
        formatted_t = torch.stack(padded_targets).to(self.device)

        return formatted_x, formatted_t

    def run_embedding_version1(self, batch_x, dirs):
        nb_streamlines = len(batch_x)

        # Packing inputs and saving info
        inputs = pack_sequence(batch_x, enforce_sorted=False).to(self.device)
        targets = pack_sequence(dirs, enforce_sorted=False).to(self.device)
        batch_sizes = inputs.batch_sizes
        sorted_indices = inputs.sorted_indices
        unsorted_indices = inputs.unsorted_indices
        nb_input_points = len(inputs.data)

        logger.debug("Preparing the {} points for this batch (total for the "
                     "{} streamlines)".format(nb_input_points, nb_streamlines))

        logger.debug("*** 1.A. Previous dir embedding, if any "
                     "(on packed_sequence's tensor!)...")
        n_prev_dirs_embedded = super().run_prev_dirs_embedding_layer(
            dirs, unpack_results=False)\

        logger.debug("*** 1.B. Inputs embedding (on packed_sequence's "
                     "tensor!)...")
        logger.debug("Nb features per point: {}".format(inputs.data.shape[-1]))
        inputs = self.embedding_layer_x(inputs.data)
        logger.debug("Embedded size: {}".format(inputs.shape[-1]))

        logger.debug("*** 1.C. Targets embedding (on packed_sequence's "
                     "tensor!)...")
        logger.debug("Target (3 coords): {}".format(targets.data.shape[-1]))
        targets = self.embedding_layer_t(targets.data)
        logger.debug("Target embedded size: {}".format(targets.shape[-1]))
        # Unpacking
        targets = PackedSequence(targets, batch_sizes, sorted_indices,
                                 unsorted_indices)
        targets = unpack_sequence(targets)

        logger.debug("*** 2. Concatenating previous dirs and inputs's "
                     "embeddings...")
        if n_prev_dirs_embedded is not None:
            inputs = torch.cat((inputs, n_prev_dirs_embedded), dim=-1)
            logger.debug("Concatenated shape: {}".format(inputs.shape))
        logger.debug("Input final size: {}".format(inputs.shape[-1]))
        # Unpacking
        inputs = PackedSequence(inputs, batch_sizes, sorted_indices,
                                unsorted_indices)
        inputs = unpack_sequence(inputs)

        logger.debug("*** 3. Padding and arranging...")
        inputs, targets = self._pad_and_stack_batch(inputs, targets)

        logger.debug("*** 4. Positional encoding")
        inputs += self.embedding_layer_position(inputs.size(0))
        targets += self.embedding_layer_position(targets.size(0))

        return inputs, targets

    def run_embedding_version2(self, batch_x, batch_t):

        assert len(batch_x) == len(batch_t), \
            "Error, the batch does not contain the same number of inputs " \
            "and targets..."

        logger.debug("*** 1. Padding and arranging.")

        formatted_x, formatted_y = self._pad_and_stack_batch(padded_inputs,
                                                             padded_targets)

        logger.debug("*** 2. Concatenating")


        logger.debug("*** 3 and 4: Embedding + positional encoding.")
        embed_x = self.embedding_layer_x(formatted_x) + \
            self.embedding_layer_position(formatted_x)

        # Embedding targets
        embed_t = self.embedding_layer_t(formatted_t) + \
            self.embedding_layer_position(formatted_t)

        return embed_x, embed_t

    def _prepare_masks_batch(self, batch_x):
        future_mask = Transformer.generate_square_subsequent_mask(self.max_len)

        batch_padded_masks = torch.zeros(len(batch_x), self.max_len)
        for i in range(len(batch_x)):
            x = batch_x[i]
            batch_padded_masks[i, len(x):-1] = float('-inf')

        return future_mask, batch_padded_masks

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
    norm_first = False  # If True, encoder and decoder layers will perform

    # LayerNorms before other attention and feedforward operations, otherwise
    # after. Torch default + in original paper: False.

    def __init__(self, experiment_name: str, nb_features: int,
                 # PREVIOUS DIRS
                 nb_previous_dirs: int = 0,
                 prev_dirs_embedding_size: int = None,
                 prev_dirs_embedding_key: str = None,
                 # INPUTS
                 max_len: int = 3500,
                 positional_encoding_key: str = 'sinusoidal',
                 x_embedding_key: str = 'nn_embedding',
                 t_embedding_key: str = 'nn_embedding',
                 # TRANSFORMER
                 d_model: int = 4096, dim_ffnn: int = None, nheads: int = 8,
                 dropout_rate: float = 0.1, activation: str = 'relu',
                 n_layers_e: int = 6, n_layers_d: int = 6,
                 # DIRECTION GETTER
                 dg_key: str = 'cosine-regression', dg_args: dict = None,
                 # Other
                 neighborhood_type: str = None,
                 neighborhood_radius: Union[int, float, List[float]] = None,
                 normalize_directions=True,
                 log_level=logging.root.level):
        """
        Args
        ----
        n_layers_e: int
            Number of encoding layers in the encoder. [6]
        n_layers_d: int
            Number of encoding layers in the decoder. [6]
        """
        super().__init__(experiment_name, nb_features, nb_previous_dirs,
                         prev_dirs_embedding_size, prev_dirs_embedding_key,
                         max_len, positional_encoding_key, x_embedding_key,
                         t_embedding_key, d_model, dim_ffnn, nheads,
                         dropout_rate, activation, dg_key, dg_args,
                         neighborhood_type, neighborhood_radius,
                         normalize_directions, log_level)

        # ----------- Additional params
        self.n_layers_e = n_layers_e
        self.n_layers_d = n_layers_d

        # ----------- Additional instantiations
        logger.info("Instantiating torch transformer, may take a few "
                    "seconds...")
        # Encoder:
        encoder_layer = TransformerEncoderLayer(
            self.d_model, self.nheads, self.dim_ffnn, self.dropout_rate,
            self.activation, batch_first=True, norm_first=self.norm_first)
        encoder = TransformerEncoder(encoder_layer, n_layers_e, norm=None)

        # Decoder
        decoder_layer = TransformerDecoderLayer(
            self.d_model, self.nheads, self.dim_ffnn, self.dropout_rate,
            self.activation, batch_first=True, norm_first=self.norm_first)
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

    def __init__(self, experiment_name: str, nb_features: int,
                 # PREVIOUS DIRS
                 nb_previous_dirs: int = 0,
                 prev_dirs_embedding_size: int = None,
                 prev_dirs_embedding_key: str = None,
                 # INPUTS
                 max_len: int = 3500,
                 positional_encoding_key: str = 'sinusoidal',
                 x_embedding_key: str = 'nn_embedding',
                 t_embedding_key: str = 'nn_embedding',
                 # TRANSFORMER
                 d_model: int = 4096, dim_ffnn: int = None, nheads: int = 8,
                 dropout_rate: float = 0.1, activation: str = 'relu',
                 n_layers_d: int = 6,
                 # DIRECTION GETTER
                 dg_key: str = 'cosine-regression', dg_args: dict = None,
                 # Other
                 neighborhood_type: str = None,
                 neighborhood_radius: Union[int, float, List[float]] = None,
                 normalize_directions=True,
                 log_level=logging.root.level):
        """
        Args
        ----
        n_layers_d: int
            Number of encoding layers in the decoder. [6]
        """
        super().__init__(experiment_name, nb_features, nb_previous_dirs,
                         prev_dirs_embedding_size, prev_dirs_embedding_key,
                         max_len, positional_encoding_key, x_embedding_key,
                         t_embedding_key, d_model, dim_ffnn, nheads,
                         dropout_rate, activation, dg_key, dg_args,
                         neighborhood_type, neighborhood_radius,
                         normalize_directions, log_level)

        # ----------- Additional params
        self.n_layers_d = n_layers_d

        # ----------- Additional instantiations
        # We say "decoder only" from the logical point of view, but code-wise
        # it is actually "encoder only". A decoder would need output from the
        # encoder.
        logger.debug("Instantiating Transformer...")
        double_layer = TransformerEncoderLayer(
            self.d_model * 2, self.nheads, self.dim_ffnn, self.dropout_rate,
            self.activation)
        self.main_layer = TransformerEncoder(double_layer, n_layers_d,
                                             norm=None)

    def _run_main_layer_forward(self, embed_x, embed_t, mask, padded_mask):
        # Concatenating x and t
        inputs = torch.cat((embed_x, embed_t), dim=-1)
        logger.debug("Concatenated [src | tgt] shape: {}".format(inputs.shape))

        # Doubling the masks
        double_mask = torch.cat((mask, mask), dim=-1)
        double_padded_mask = torch.cat((padded_mask, padded_mask), dim=-1)
        logger.debug("Concatenated mask shape: {}".format(inputs.shape))

        # Main transformer
        outputs = self.main_layer(src=inputs, src_mask=double_mask,
                                  src_key_padding_mask=double_padded_mask)

        # Take the second half of model outputs to direction getter
        # (the last skip-connection makes more sense this way)
        return outputs[-self.d_model:-1]
