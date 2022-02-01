# -*- coding: utf-8 -*-
import logging

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


def prepare_original_model(args):
    _perform_checks(args)
    model = OriginalTransformerModel()
    return model


def prepare_src_tgt_model(args):
    _perform_checks(args)
    model = TransformerSourceAndTargetModel()
    return model