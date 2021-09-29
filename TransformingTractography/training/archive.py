# -*- coding: utf-8 -*-

from dwi_ml.experiment.trainer_abstract import StreamlinesBasedModelAbstract

from TransformingTractography.models.embedding import (
    CompleteEmbedding, ConvolutionalDataEmbedding,
    NeuralNetworkDataEmbedding, SimpleDataEmbedding, SinusoidalPosEmbedding,
    RelationalSinusoidalPosEmbedding)
from TransformingTractography.models.transformer import OurTransformer

class Trainer(StreamlinesBasedModelAbstract):
    def __init__(self, args):
        super().__init__(args)

        self.nheads = args.nheads
        self.nb_classes = args.nb_classes
        self.dropout = args.dropout_rate

        # Embedding:
        self.d_model = args.embedding_size
        self.data_embedding_choice = args.data_embedding
        self.position_embedding_choice = args.position_embedding
        if args.hidden_layers_NN:
            self.nb_hidden_layers_NN = args.hidden_layers_NN[0]
            self.hidden_layers_size_NN = args.hidden_layers_NN[1]
        else:
            self.nb_hidden_layers_NN = 0
            self.hidden_layers_size_NN = 0
        self.max_seq = args.max_seq

        # Encoding/Decoding:
        self.n_layers_e = args.n_layers_e
        self.n_layers_d = args.n_layers_d

        # Data. Should be defined in super
        # self.data_x
        # self.data_y
        # self.input_size.

        # Model. To be defined with build_model
        self.transformer = None

    def load_dataset(self):
        super().load_dataset()

    def build_model(self):
        """
        Build PyTorch Transformer models.
        """
        data_embedding_builder = {
            'simple': SimpleDataEmbedding(), #toDO
            'ffnn': NeuralNetworkDataEmbedding(
                input_size=self.input_size, d_model=self.d_model,
                size_hidden_layers=self.hidden_layers_size_NN,
                nb_hidden_layers=self.nb_hidden_layers_NN),
            'cnn': ConvolutionalDataEmbedding() #toDO
        }
        position_embedding_builder = {
            'sinusoidal': SinusoidalPosEmbedding(d_model=self.d_model,
                                                 max_seq=self.max_seq),
            'relational': RelationalSinusoidalPosEmbedding() #toDo
        }

        # Embedding
        data_embedding_layer_x = \
            data_embedding_builder(self.data_embedding_choice_x)
        position_embedding_layer_x = \
            position_embedding_builder(self.position_embedding_choice_x)
        embedding_layer_x = CompleteEmbedding(self.d_model,
                                                 data_embedding_layer,
                                                 position_embedding_layer,
                                                 self.dropout)
        embedding_layer_y = None #ToDo

        # Complete models
        self.transformer = OurTransformer(embedding_layer_x, embedding_layer_y,
                                          self.nb_classes, self.d_model,
                                          self.nheads, self.n_layers_e,
                                          self.n_layers_d, self.dim_ffnn,
                                          self.dropout, self.activation)

    def train(self, **kwargs):
        self.transformer(self.data_x, self.data_y)


    def save(self):
        super().save()



