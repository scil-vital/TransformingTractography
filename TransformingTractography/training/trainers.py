# -*- coding: utf-8 -*-
from dwi_ml.models.batch_samplers import (
    BatchStreamlinesSampler1IPV as BatchSampler)
from Learn2Track.training.trainers import Learn2TrackTrainer

from TransformingTractography.models.transformer import OurTransformer

# For now, copying Learn2track trainer. Let's see if anything will be different


class TransformerTrainer(Learn2TrackTrainer):
    def __init__(self,
                 batch_sampler_training: BatchSampler,
                 batch_sampler_validation: BatchSampler,
                 model: OurTransformer, experiment_path: str,
                 experiment_name: str, learning_rate: float,
                 weight_decay: float, max_epochs: int,
                 max_batches_per_epoch: int, patience: int,
                 nb_cpu_workers: int, taskman_managed: bool, use_gpu: bool,
                 comet_workspace: str, comet_project: str,
                 from_checkpoint: bool, clip_grad: float, **_):
        super().__init__(batch_sampler_training, batch_sampler_validation,
                         model, experiment_path, experiment_name,
                         learning_rate, weight_decay, max_epochs,
                         max_batches_per_epoch, patience, nb_cpu_workers,
                         taskman_managed, use_gpu, comet_workspace,
                         comet_project, from_checkpoint, clip_grad)