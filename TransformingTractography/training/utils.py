# -*- coding: utf-8 -*-
import logging

from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer


from TransformingTractography.training.trainers import TransformerTrainer


def prepare_trainer(training_batch_sampler, validation_batch_sampler,
                    training_batch_loader, validation_batch_loader,
                    model, args):
    # Instantiate trainer
    with Timer("\n\nPreparing trainer", newline=True, color='red'):
        trainer = TransformerTrainer(
            training_batch_sampler, validation_batch_sampler,
            training_batch_loader, validation_batch_loader,
            model,
            args.experiment_path, args.experiment_name,
            # COMET
            comet_project=args.comet_project,
            comet_workspace=args.comet_workspace,
            # TRAINING
            learning_rate=args.learning_rate, max_epochs=args.max_epochs,
            max_batches_per_epoch=args.max_batches_per_epoch,
            patience=args.patience, from_checkpoint=False,
            weight_decay=args.weight_decay,
            # MEMORY
            nb_cpu_processes=args.processes,
            taskman_managed=args.taskman_managed, use_gpu=args.use_gpu)
        logging.info("Trainer params : " + format_dict_to_str(trainer.params))

    return trainer
