# -*- coding: utf-8 -*-
import logging

from TransformingTractography.models.positional_embeddings import (
    DATA_EMBEDDING_CHOICES, POSITION_EMBEDDING_CHOICES)
from TransformingTractography.models.transformer import \
    TransformingTractographyModel


def add_model_args(p):
    """ Optional parameters for TransformingTractography"""
    gx = p.add_argument_group(
        title="Embedding X. Total embedding will be : \n"
              "(data_embedding + position_embedding) * "
              "sqrt(embedding_size) + dropout")

    # Note about embedding. In [1], word2vec. Embedding to a fixed vocabulary
    # size. torch.nn.Embedding(dict_size??, self.d_model). Not really feasible
    # here.
    gx.add_argument(
        '--data_embedding', default='simple',
        choices=DATA_EMBEDDING_CHOICES.keys(),
        help="Type of data embedding to use. Default: [%(default)s]\n"
             "   -Simple: Like word2vec. Needs a fixed vocabulary size. \n"
             "   -ffnn: Learns an embedding with a simple neural network. No "
             "fixed vocabulary size.\n"
             "   -cnn: Learns an embedding with a convolutional neural "
             "network (CNN). Not ready. Args to be added.")
    gx.add_argument(
        '--data_embedding_size', type=int, default=512,
        help="Embedding size, and thus size of the input the transformer "
             "will receive. (d_model in [1], D in [2]).\n "
             "Value in [1] and [2]: 512. Default: [%(default)s]")
    gx.add_argument(
        '--hidden_layers_NN', type=[int, int],
        metavar='[NB_HIDDEN_LAYERS, SIZE_HIDDEN_LAYERS]',
        help="Number of hidden layers in the case of neural network data "
             "embedding, and their size (the way we parepared this, they must "
             "all have the same size. Default: No hidden layers.")

    gx.add_argument(
        '--position_embedding', default='sinusoidal',
        choices=POSITION_EMBEDDING_CHOICES.keys(),
        help="Type of positional embedding to use. Default: [%(default)s]\n"
             "   -Sinusoidal: Such as described in [1]. Also used in [2].\n"
             "   -Relational: Such as described in [2].")
    gx.add_argument(
        '--max_seq', type=int,
        help="Longest sequence allowed. Only necessary, but then mandatory, "
             "with sinusoidal position embedding.\n"
             "Value in [1]: ?. In [2]: 3500.")

    gy = p.add_argument_group(title='Embedding Y. ToDo')
    gy.add_argument(
        '--nb_classes', type=int, default=30,
        help="Number of classes as output choices. Default: [%(default)s]")

    gt = p.add_argument_group(title='Transformer')
    gt.add_argument(
        '--n_layers_e', type=int, default=6,
        help="Number of encoding layers. Value in [1] and [2]; 6. "
             "Default: [%(default)s]")
    gt.add_argument(
        '--n_layers_d', type=int, default=6,
        help="Number of decoding layers. Value in [1] and [2]; 6. "
             "Default: [%(default)s]")
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


def prepare_model(args):
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

    model = TransformingTractographyModel(

    )
