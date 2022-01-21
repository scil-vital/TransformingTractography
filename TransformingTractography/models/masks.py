# -*- coding: utf-8 -*-
# See here for examples and explanations
# https://stackoverflow.com/questions/62170439/difference-between-src-mask-and-src-key-padding-mask

"""
To verify mask use: During forward:

    transformer:
        calls encoder:
            loops on layers and calls encoderLayer:
                calls _sa_block:
                    calls self.self_attn = MultiHeadAttention(src_mask)

        calls decoder:
            loops on layers and calls decoderLayer:
                calls _sa_block:
                    calls self.self_attn = MultiHeadAttention(tgt_mask)

In our case, length of src = length of tgt so
src_mask = tgt_mask = memory_mask = LxL!

-------------------
Note: We have two reasons for masks: hide the padded length +
                                     hide the future in generation.

Typically, in the MultiHeadAttention, hidding paddded time points is done by
hiding the keys only through key_padding_mask and others through attn_mask.
In practice, what they do is attn_mask = attn_mask.logical_or(key_padding_mask)
so it really doesn't change anything.

We actually simply use one mask for both use. In all case, it means we hide
all padded data, either padded because streamline is short because it is not
fully generated yet during tracking or because streamline is simply short
during training.
"""
import torch


def prepare_masks_batch(batch_x, padded_length, nheads):
    all_masks = []
    for i in range(len(batch_x)):
        x = batch_x[i]

        # src_mask = tgt_mask = memory_mask = LxL!
        # we hide the padded points.
        all_masks.append(prepare_masks_one_input(padded_length, len(x)))

    # Make a batch of masks
    # Final size should be N*L*L
    batch_masks = torch.cat(all_masks, dim=0)

    # Then we must make this of size (N * nheads, L, S)
    batch_masks = batch_masks.repeat(nheads, 1, 1)
    return batch_masks


def prepare_masks_one_input(padded_length, actual_current_length):
    """
    Params
    ------
    padded_length = L = length of the sequences (padded length).

    """
    # Generate a square mask for the sequence.
    # The masked positions are filled with float('-inf').
    # Unmasked positions are filled with float(0.0).

    # toDo check which points we hide. From actual_current_length forward, or
    #  from actual_current_length + 1??
    self_mask = torch.triu(torch.full((padded_length, padded_length),
                                      float(0.0)))
    self_mask[actual_current_length:-1, actual_current_length:-1] = \
        float('-inf')

    return self_mask
