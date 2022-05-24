# -*- coding: utf-8 -*-
import logging

import torch
from torch.nn.utils.rnn import pack_sequence

from TransformingTractography.models.transformer import (
    OriginalTransformerModel)


def _create_batch():
    logging.debug("Creating batch: 2 streamlines, the first has 4 points "
                  "and the second, 3. Input: 4 features per point.")

    # dwi1 : data for the 3 first points
    flattened_dwi1 = [[10., 11., 12., 13.],
                      [50., 51., 52., 53.],
                      [60., 62., 62., 63.]]

    # dwi2 : data for the 2 first points
    flattened_dwi2 = [[10., 11., 12., 13.],
                      [50., 51., 52., 53.]]

    batch_x = [torch.Tensor(flattened_dwi1), torch.Tensor(flattened_dwi2)]

    return batch_x


def test_original_model():
    batch_x = _create_batch()
    batch_x_packed = pack_sequence(batch_x, enforce_sorted=False)

    model = OriginalTransformerModel('test', )

    # Model's logger level can be set by using the logger's name.
    logger = logging.getLogger('model_logger')
    logger.setLevel('DEBUG')

    # Testing forward.
    output, _hidden_state = model(batch_x_packed)

    assert len(output) == 5  # Total number of points.
    assert output.shape[1] == 6  # 3 + 3 with skip connections


if __name__ == '__main__':
    logging.basicConfig(level='DEBUG')
    test_original_model()

