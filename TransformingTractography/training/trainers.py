# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from dwi_ml.training.batch_samplers import DWIMLBatchSampler
from dwi_ml.training.batch_loaders import BatchLoaderOneInput
from dwi_ml.training.trainers import DWIMLTrainerOneInput

from TransformingTractography.models.transformer import \
    AbstractTransformerModel


class TransformerTrainer(DWIMLTrainerOneInput):
    def __init__(self,
                 batch_sampler_training: DWIMLBatchSampler,
                 batch_sampler_validation: DWIMLBatchSampler,
                 batch_loader_training: BatchLoaderOneInput,
                 batch_loader_validation: BatchLoaderOneInput,
                 model: AbstractTransformerModel, experiment_path: str,
                 experiment_name: str, learning_rate: float,
                 weight_decay: float, max_epochs: int,
                 max_batches_per_epoch: int, patience: int,
                 nb_cpu_processes: int, taskman_managed: bool, use_gpu: bool,
                 comet_workspace: str, comet_project: str,
                 from_checkpoint: bool):
        super().__init__(batch_sampler_training, batch_sampler_validation,
                         batch_loader_training, batch_loader_validation,
                         model, experiment_path, experiment_name,
                         learning_rate, weight_decay, max_epochs,
                         max_batches_per_epoch, patience,
                         nb_cpu_processes, taskman_managed, use_gpu,
                         comet_workspace, comet_project,
                         from_checkpoint)

    # Batch size:
    # Copying this from learn2track. many todos. Let's see if it gets
    # modified.
    def estimate_nb_batches_per_epoch(self):
        logging.info("Learn2track: Estimating training epoch statistics...")
        n_train_batches_capped, _ = self._estimate_nb_batches_per_epoch(
            self.train_batch_sampler, self.train_batch_loader)

        n_valid_batches_capped = None
        if self.valid_batch_sampler is not None:
            logging.info("Learn2track: Estimating validation epoch "
                         "statistics...")
            n_valid_batches_capped, _ = self._estimate_nb_batches_per_epoch(
                self.valid_batch_sampler, self.valid_batch_loader)

        return n_train_batches_capped, n_valid_batches_capped

    def _estimate_nb_batches_per_epoch(self, batch_sampler: DWIMLBatchSampler,
                                       batch_loader: BatchLoaderOneInput):
        """
        Compute the number of batches necessary to use all the available data
        for an epoch (but limiting this to max_nb_batches).

        Returns
        -------
        n_batches : int
            Approximate number of updates per epoch
        batch_size : int
            Batch size or approximate batch size.
        """
        # Here, 'batch_size' will be computed in terms of number of
        # streamlines.
        dataset_size = batch_sampler.dataset.total_nb_streamlines[
            batch_sampler.streamline_group_idx]

        if batch_sampler.batch_size_units == 'nb_streamlines':
            # Then the batch size may actually be different, if some
            # streamlines were split during data augmentation. But still, to
            # use all the data in one epoch, we simply need to devide the
            # dataset_size by this:
            batch_size = batch_sampler.batch_size
        else:  # batch_sampler.batch_size_units == 'length_mm':
            # Then the batch size is more or less exact (with the added
            # gaussian noise possibly changing this a little bit but not much).
            # But we don't know the actual size in number of streamlines.
            raise NotImplementedError

        # Define the number of batches per epoch
        n_batches = int(dataset_size / batch_size)
        n_batches_capped = min(n_batches, self.max_batches_per_epochs)

        logging.info("Dataset had {} streamlines (before data augmentation) "
                     "and each batch contains ~{} streamlines.\nWe will be "
                     "using approximately {} batches per epoch (but not more "
                     "than the allowed {}).\n"
                     .format(dataset_size, batch_size, n_batches,
                             self.max_batches_per_epochs))

        return n_batches_capped, batch_size

    @classmethod
    def init_from_checkpoint(
            cls, train_batch_sampler: DWIMLBatchSampler,
            valid_batch_sampler: DWIMLBatchSampler,
            train_batch_loader: BatchLoaderOneInput,
            valid_batch_loader: BatchLoaderOneInput,
            model: AbstractTransformerModel,
            checkpoint_state: dict, new_patience, new_max_epochs):
        """
        During save_checkpoint(), checkpoint_state.pkl is saved. Loading it
        back offers a dict that can be used to instantiate an experiment and
        set it at the same state as previously. (Current_epoch is updated +1).
        """

        # Use super's method but return this transformer trainer as 'cls'.
        experiment = super(cls, cls).init_from_checkpoint(
            train_batch_sampler, valid_batch_sampler,
            train_batch_loader, valid_batch_loader, model,
            checkpoint_state, new_patience, new_max_epochs)

        return experiment

    def run_model(self, batch_inputs, batch_streamlines):
        model_outputs = self.model(batch_inputs, batch_streamlines)
        return model_outputs
