# -*- coding: utf-8 -*-
import logging

from dwi_ml.data.processing.space.neighborhood import add_args_neighborhood
from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.models.embeddings_on_tensors import keys_to_embeddings
from dwi_ml.models.utils.direction_getters import check_args_direction_getter

from TransformingTractography.models.positional_encoding import (
    keys_to_positional_encodings)
from TransformingTractography.models.transformer import \
    OriginalTransformerModel, TransformerSourceAndTargetModel


def add_abstract_model_args(p):
    """ Optional parameters for TransformingTractography"""
    gx = p.add_argument_group("Embedding:")
    gx.add_argument(
        '--data_embedding', default='nn_embedding',
        choices=keys_to_embeddings.keys(), metavar='key',
        help="Type of data embedding to use. One of 'no_embedding', \n"
             "'nn_embedding' (default) or 'cnn_embedding'.")
    gx.add_argument(
        '--position_encoding', default='sinusoidal', metavar='key',
        choices=keys_to_positional_encodings.keys(),
        help="Type of positional embedding to use. One of 'sinusoidal' "
             "(default)\n or 'relational'. ")
    gx.add_argument(
        '--target_embedding', default='nn_embedding',
        choices=keys_to_embeddings.keys(), metavar='key',
        help="Type of data embedding to use. One of 'no_embedding', \n"
             "'nn_embedding' (default) or 'cnn_embedding'.")

    gt = p.add_argument_group(title='Transformer')
    gt.add_argument(
        '--d_model', type=int, default=4096, metavar='n',
        help="Output size that will kept constant in all layers to allow \n"
             "skip connections (embedding size, ffnn output size, attention \n"
             "size). [%(default)s]")
    gt.add_argument(
        '--max_len', type=int, default=1000, metavar='n',
        help="Longest sequence allowed. Other sequences will be zero-padded \n"
             "up to that length (but attention can't attend to padded "
             "timepoints).\nPlease beware that this value influences strongly "
             "the executing time and heaviness.\nAlso used with sinusoidal "
             "position embedding. [%(default)s]")
    gt.add_argument(
        '--nheads', type=int, default=8, metavar='n',
        help="Number of heads per layer. Could be different for each layer \n"
             "but we decided not to implement this possibility. [%(default)s]")
    gt.add_argument(
        '--dropout_rate', type=float, default=0.1, metavar='r',
        help="Dropout rate for all dropout layers. Again, could be different\n"
             "in every layers but that's not the choice we made.\n"
             "Needed in embedding, encoder and decoder. [%(default)s]")
    gt.add_argument(
        '--ffnn_hidden_size', type=int, default=None, metavar='n',
        help="Size of the feed-forward neural network (FFNN) layer in the \n"
             "encoder and decoder layers. The FFNN is composed of two linear\n"
             "layers. This is the size of the output of the first one. \n"
             "Default: data_embedding_size/2")
    gt.add_argument(
        '--activation', choices=['relu', 'gelu'], default='relu',
        metavar='key',
        help="Choice of activation function in the FFNN. One of 'relu' or \n"
             "'gelu'. [%(default)s]")

    g = p.add_argument_group("Preprocessing")
    g.add_argument(
        '--normalize_directions', action='store_true',
        help="If true, directions will be normalized. If the step size is \n"
             "fixed, it shouldn't make any difference. If streamlines are \n"
             "compressed, in theory you should normalize, but you could hope\n"
             "that not normalizing could give back to the algorithm a sense \n"
             "of distance between points.")
    add_args_neighborhood(g)

    return gt


def add_original_model_args(gt):
    gt.add_argument(
        '--n_layers_e', type=int, default=6,
        help="Number of encoding layers. [%(default)s]")
    gt.add_argument(
        '--n_layers_d', type=int, default=6,
        help="Number of decoding layers. [%(default)s]")


def add_src_tgt_attention_args(gt):
    gt.add_argument(
        '--n_layers_d', type=int, default=14,
        help="Number of 'decoding' layers. Value in [3]; 14. "
             "Default: [%(default)s].\n"
             "[3]: https://arxiv.org/pdf/1905.06596.pdf")


def perform_checks(args):
    # Deal with your optional parameters:
    if args.dropout_rate < 0 or args.dropout_rate > 1:
        raise ValueError('The dropout rate must be between 0 and 1.')

    if not args.ffnn_hidden_size:
        args.ffnn_hidden_size = int(args.d_model/ 2)

    # Prepare args for the direction getter
    if not args.dg_dropout and args.dropout_rate:
        args.dg_dropout = args.dropout_rate
    dg_args = check_args_direction_getter(args)

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

    return args, dg_args


def prepare_original_model(args, dg_args):
    with Timer("\n\nPreparing model", newline=True, color='yellow'):
        model = OriginalTransformerModel(
            experiment_name=args.experiment_name,
            # Concerning inputs:
            neighborhood_type=args.neighborhood_type,
            neighborhood_radius=args.neighborhood_radius,
            nb_features=args.nb_features,
            # Concerning embedding:
            max_len=args.max_len,
            positional_encoding_key=args.position_encoding,
            x_embedding_key=args.data_embedding,
            t_embedding_key=args.target_embedding,
            # Torch's transformer parameters
            d_model=args.d_model, dim_ffnn=args.ffnn_hidden_size,
            nheads=args.nheads, dropout_rate=args.dropout_rate,
            activation=args.activation, n_layers_e=args.n_layers_e,
            n_layers_d=args.n_layers_e,
            # Direction getter
            dg_key=args.dg_key, dg_args=dg_args,
            normalize_directions=args.normalize_directions)

        logging.info("Transformer (original) model final parameters:" +
                     format_dict_to_str(model.params))

    return model


def prepare_src_tgt_model(args):
    dg_args, args = perform_checks(args)

    with Timer("\n\nPreparing model", newline=True, color='yellow'):
        model = TransformerSourceAndTargetModel(
            experiment_name=args.experiment_name,
            # Concerning inputs:
            neighborhood_type=args.neighborhood_type,
            neighborhood_radius=args.neighborhood_radius,
            nb_features=args.nb_features,
            # Concerning embedding:
            max_len=args.max_len,
            positional_encoding_key=args.position_encoding,
            x_embedding_key=args.data_embedding,
            t_embedding_key=args.target_embedding,
            # Torch's transformer parameters
            d_model=args.d_model, dim_ffnn=args.ffnn_hidden_size,
            nheads=args.nheads, dropout_rate=args.dropout_rate,
            activation=args.activation,
            n_layers_d=args.n_layers_e,
            # Direction getter
            dg_key=args.dg_key, dg_args=dg_args,
            normalize_directions=args.normalize_directions)

        logging.info("Transformer (src-tgt attention) model final "
                     "parameters:" +
                     format_dict_to_str(model.params))

    return model
