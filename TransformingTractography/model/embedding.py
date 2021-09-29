# -*- coding: utf-8 -*-

"""
Data embedding:
    - SimpleDataEmbedding
    - NeuralNetworkDataEmbedding
    - ConvolutionalDataEmbedding

Positional embeddings:
    - SinusoidalPosEmbedding
    - RelationalSinusoidalPosEmbedding
"""

import math as m
import numpy as np
import copy

import torch
from torch.nn.modules.linear import Linear
from torch.nn import ModuleList


class SimpleDataEmbedding(torch.nn.Module):
    """Like word2vec. Needs a fixed vocabulary size."""
    def __init__(self):
        raise NotImplementedError
        # See torch.nn.Embedding(dict_size??, self.d_model)


class NeuralNetworkDataEmbedding(torch.nn.Module):
    """
    Learns an embedding with a simple neural network. Purpose: in the case where
    your data is already a vector (ex, values in dWI), this is not like learning
    a word2vec with a fixed vocabulary size. We could even use the raw data,
    technically. But when adding the positional embedding, the reason it works
    is that the learning of the embedding happens while knowing that some
    positional vector will be added to it. As stated in the blog
    https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    the embedding probably adapts to leave place for the positional embedding.

    All this to say we can't use raw data, and the minimum is to learn to adapt
    with a neural network.
    """
    def __init__(self, input_size, d_model, size_hidden_layers,
                 nb_hidden_layers):
        super().__init__()

        # Save parameters
        self.input_size = input_size
        self.output_size = d_model
        self.size_hidden_layers = size_hidden_layers
        self.nb_hidden_layers = nb_hidden_layers
        bias = True

        if nb_hidden_layers>=1:
            # Instantiate layers
            first_layer = Linear(input_size, size_hidden_layers, bias)
            last_layer = Linear(size_hidden_layers,d_model, bias)
            if nb_hidden_layers==1:
                self.layers = [first_layer, last_layer]
            else:
                middle_layer =  Linear(input_size, size_hidden_layers, bias)
                middle_layers = [copy.deepcopy(middle_layer)
                                for i in range(nb_hidden_layers)]
                self.layers = [first_layer]
                self.layers.extend(middle_layers)
                self.layers.append(last_layer)
        else:
            self.layers = [Linear(input_size, d_model, bias)]

    def forward(self, x):
        for i in range(self.nb_hidden_layers+1):
            x = self.layers[i](x)

        return x


class ConvolutionalDataEmbedding(torch.nn.Module):
    """Uses a CNN to train the data"""
    def __init__(self):
        raise NotImplementedError


class SinusoidalPosEmbedding(torch.nn.Module):
    """
    Creates the vector for positional embedding that can be added to the data
    embedding as first layer of a transformer. Such as described in the paper
    Attention is all you need.
    """
    def __init__(self, d_model: int, max_seq: int = 2048):
        super().__init__()

        self.d_emb = d_model
        self.max_seq = max_seq

    def forward(self):
        """
        Using formula from paper for odd and even positions
        """
        pos_emb = np.zeros((self.max_seq, self.d_emb))
        for index in range(0, self.d_emb, 2):
            pos_emb[:, index] = np.array(
                [m.sin(pos / 10000 ** (index / self.d_emb))
                 for pos in range(self.max_seq)])
            pos_emb[:, index + 1] = np.array(
                [m.cos(pos / 10000 ** (index / self.d_emb))
                 for pos in range(self.max_seq)])

        return pos_emb


class RelationalSinusoidalPosEmbedding(torch.nn.Module):
    """
    Creates the vector for positional embedding that can be added to the data
    embedding as first layer of a transformer. Such as described in the music
    paper to replace the sinusoidal version.
    """
    def __init__(self):
        super().__init__()
        raise NotImplementedError


class CompleteEmbedding(torch.nn.Module):
    """
    1    Chosen data embedding
    2  + Chosen positional embedding
    3  * sqrt(d_model)                                                                                                # ??? Expliquent pas pourquoi...
    4  + dropout.

    Layers are instantiated in __init__ and combined together in forward.
    """
    def __init__(self, d_model: int, data_emb_layer: torch.nn.Module,
                 pos_emb_layer: torch.nn.Module, dropout: float = 0.1):
        """
        emb_layer: torch.nn.Module
            A pre-instantiated torch module that will act as embedding. We will
            add position embedding to the result. Suggestion: torch.nn.Embedding
            or a more complicated structure, such as a CNN. The layer must
            create an output vector of length d_model.
        pos_emb_choice: str
            Choose between 'sinusoidal' or 'relational' for the position
            embedding ['sinusoidal'].
        """
        super().__init__()

        self.data_embedding_layer = data_emb_layer
        self.positional_embedding_layer = pos_emb_layer
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, d_emb: int, dropout: float = 0.1):
        """
        Dropout((embedding + positional embedding) * sqrt(d_emb))
        """
        x = self.data_embedding_layer(x) + self.positional_embedding_layer(x)
        x *= np.sqrt(d_emb)                                                                                     # ToDo. Ils font ça pcq ils sharent the weights of embedding layers and decoder.
                                                                                                                #  Pour l'instant on ne share pas donc peut-être pas nécessaire.
        x = self.dropout(x)

        return x


DATA_EMBEDDING_CHOICES = {
    'simple': SimpleDataEmbedding,
    'ffnn': NeuralNetworkDataEmbedding,
    'cnn': ConvolutionalDataEmbedding
}

POSITION_EMBEDDING_CHOICES = {
    'sinusoidal': SinusoidalPosEmbedding,
    'relational': RelationalSinusoidalPosEmbedding,
}