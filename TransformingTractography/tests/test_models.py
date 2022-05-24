# -*- coding: utf-8 -*-
import logging

import torch
from torch.nn.utils.rnn import pack_sequence

from TransformingTractography.models.transformer import (
    OriginalTransformerModel, TransformerSourceAndTargetModel)


def _create_batch():
    logging.debug("Creating batch: 2 streamlines, the first has 4 points "
                  "and the second, 3. Input: 4 features per point.")

    # dwi1 : data for the 3 first points
    flattened_dwi1 = [[10., 11., 12., 13.],
                      [50., 51., 52., 53.],
                      [60., 62., 62., 63.]]
    streamline1 = [[0.1, 0.2, 0.3],
                   [1.1, 1.2, 1.3],
                   [2.1, 2.2, 2.3],
                   [3.1, 3.2, 3.3]]

    # dwi2 : data for the 2 first points
    flattened_dwi2 = [[10., 11., 12., 13.],
                      [50., 51., 52., 53.]]
    streamline2 = [[10.1, 10.2, 10.3],
                   [11.1, 11.2, 11.3],
                   [12.1, 12.2, 12.3]]

    batch_x = [torch.Tensor(flattened_dwi1), torch.Tensor(flattened_dwi2)]
    batch_s = [streamline1, streamline2]

    return batch_x, batch_s


def test_models():
    batch_x, batch_streamlines = _create_batch()

    logging.debug("Original model!\n"
                  "-----------------------------")
    model = OriginalTransformerModel('test', nb_features=4,
                                     log_level='DEBUG')

    # Testing forward.
    #output, _hidden_state = model(batch_x, batch_streamlines)

    #assert len(output) == 5  # Total number of points.
    #assert output.shape[1] == 6  # 3 + 3 with skip connections

    logging.debug("Source and target model!\n"
                  "-----------------------------")
    model = TransformerSourceAndTargetModel('test', nb_features=4,
                                            log_level='DEBUG')


if __name__ == '__main__':
    logging.basicConfig(level='DEBUG')
    test_models()
