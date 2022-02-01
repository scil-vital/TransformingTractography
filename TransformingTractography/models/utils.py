# -*- coding: utf-8 -*-
import logging

from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.models.embeddings_on_tensors import keys_to_embeddings

from TransformingTractography.models.positional_encoding import (
    keys_to_positional_encodings)
from TransformingTractography.models.transformer import \
    OriginalTransformerModel, TransformerSourceAndTargetModel


def add_general_model_args(p):
    """ Optional parameters for TransformingTractography"""
    gx = p.add_argument_group("Embedding X:")
    gx.add_argument(
        '--data_embedding', default='simple',
        choices=keys_to_embeddings.keys(),
        help="Type of data embedding to use. Currently, only one layer of"
             "NN is implemented. #todo.")
    gx.add_argument(
        '--position_embedding', default='sinusoidal',
        choices=keys_to_positional_encodings.keys(),
        help="Type of positional embedding to use. Default: [%(default)s]\n"
             "   -Sinusoidal: Such as described in [1]. Also used in [2].\n"
             "   -Relational: Such as described in [2].")
    gx.add_argument(
        '--max_seq', type=int,
        help="Longest sequence allowed. Only necessary, but then mandatory, "
             "with sinusoidal position embedding.\n"
             "Value in [1]: ?. In [2]: 3500.")

    gt = p.add_argument_group(title='Transformer')
    gt.add_argument(
        '--nheads', type=int, default=8,
        help="Number of heads per layer. Could be different for each layer "
             "but we decided not to implement this possibility. Value in [1] "
             "and [2]: 8. Default: [%(default)s]")
    gt.add_argument(
        '--dropout_rate', type=float, default=0.1,
        help="Dropout rate for all the dropbout layers. Again, could be "
             "different in every layers but that's not the choice we made.\n"
             "Needed in embedding, encoder and decoder. Value in [1] and "
             "[2]: 0.1. Default: [%(default)s]")
    gt.add_argument(
        '--ffnn_size', type=int, default=None,
        help="Size of the feed-forward neural network (FFNN) layer in the "
             "encoder and decoder layers. The FFNN is composed of two linear "
             "layers. This is the size of the output of the first one. "
             "Default: data_embedding_size/2")
    gt.add_argument(
        '--activation', choices=['relu', 'gelu'], default='relu',
        help="Choice of activation function in the FFNN. Default: "
             "[%(default)s]")
    return gt


def add_original_model_args(gt):
    gt.add_argument(
        '--n_layers_e', type=int, default=6,
        help="Number of encoding layers. Value in [1] and [2]; 6. "
             "Default: [%(default)s]")
    gt.add_argument(
        '--n_layers_d', type=int, default=6,
        help="Number of decoding layers. Value in [1] and [2]; 6. "
             "Default: [%(default)s]")


def add_src_tgt_attention_args(gt):
    gt.add_argument(
        '--n_layers_d', type=int, default=14,
        help="Number of 'decoding' layers. Value in [3]; 14. "
             "Default: [%(default)s].\n"
             "[3]: https://arxiv.org/pdf/1905.06596.pdf")


def _perform_checks(args):
    # Deal with your optional parameters:
    if args.dropout_rate < 0 or args.dropout_rate > 1:
        raise ValueError('The dropout rate must be between 0 and 1.')
    if not args.ffnn_size:
        args.ffnn_size = int(args.data_embedding_size / 2)

    # check embedding choices.
    if args.position_embedding == 'sinusoidal':
        if ~args.max_seq:
            raise ValueError("Sinusoidal embedding was chosen. Please define "
                             "--max_seq.")
    else:
        if args.max_seq:
            logging.warning("--max_seq was defined but embedding choice was "
                            "not sinusoidal. max_seq is thus ignored.")

    # Prepare args for the direction getter
    dg_dropout = args.dg_dropout if args.dg_dropout else \
        args.dropout if args.dropout else 0
    dg_args = {'dropout': dg_dropout}
    if args.direction_getter_key == 'gaussian-mixture':
        nb_gaussians = args.nb_gaussians if args.nb_gaussians else 3
        dg_args.update({'nb_gaussians': nb_gaussians})
    elif args.nb_gaussians:
        logging.warning(
            "You have provided a value for --nb_gaussians but the "
            "chosen direction getter is not the gaussian mixture."
            "Ignored.")
    if args.direction_getter_key == 'fisher-von-mises-mixture':
        nb_clusters = args.nb_clusters if args.nb_nb_clusters else 3
        dg_args.update({'n_cluster': nb_clusters})
    elif args.nb_clusters:
        logging.warning(
            "You have provided a value for --nb_clusters but the "
            "chosen direction getter is not the Fisher von Mises "
            "mixture. Ignored.")

    # Prepare args for the neighborhood
    if args.grid_radius:
        args.neighborhood_radius = args.grid_radius
        args.neighborhood_type = 'grid'
    elif args.sphere_radius:
        args.neighborhood_radius = args.sphere_radius
        args.neighborhood_type = 'axes'
    else:
        args.neighborhood_radius = None
        args.neighborhood_type = None

    return dg_args, args


def prepare_original_model(args):
    dg_args, args = _perform_checks(args)

    with Timer("\n\nPreparing model", newline=True, color='yellow'):
        model = OriginalTransformerModel(
            args.experiment_name, args.neighborhood_type,
            args.neighborhood_radius, args.nb_features,
            args.padding_length, args.positional_encoding_key,
            args.x_embedding_key, args.t_embedding_key, args.d_model,
            args.dim_ffnn, args.nheads, args.dropout_rate, args.activation,
            args.n_layers_e, args.n_layers_d,
            args.direction_getter_key, dg_args, args.normalize_directions)

        logging.info("Transformer (original) model final parameters:" +
                     format_dict_to_str(model.params_per_layer))

    return model


def prepare_src_tgt_model(args):
    dg_args, args = _perform_checks(args)

    with Timer("\n\nPreparing model", newline=True, color='yellow'):
        model = TransformerSourceAndTargetModel(
            args.experiment_name, args.neighborhood_type,
            args.neighborhood_radius, args.nb_features,
            args.padding_length, args.positional_encoding_key,
            args.x_embedding_key, args.t_embedding_key, args.d_model,
            args.dim_ffnn, args.nheads, args.dropout_rate, args.activation,
            args.n_layers_d, args.direction_getter_key, dg_args,
            args.normalize_directions)

        logging.info("Transformer (src-tgt attention) model final "
                     "parameters:" +
                     format_dict_to_str(model.params_per_layer))

    return model
