#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import tempfile

from dwi_ml.tests.expected_values import TEST_EXPECTED_VOLUME_GROUPS, \
    TEST_EXPECTED_STREAMLINE_GROUPS, TEST_EXPECTED_SUBJ_NAMES
from dwi_ml.tests.utils import fetch_testing_data

data_dir = fetch_testing_data()
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('tt_train_original_model.py', '--help')
    assert ret.success

    #ret = script_runner.run('l2t_resume_training_from_checkpoint.py', '--help')
    #assert ret.success

    #ret = script_runner.run('l2t_track_from_model.py', '--help')
    #assert ret.success


def test_execution_bst(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))

    experiments_path = tmp_dir.name
    experiment_name = 'test_experiment'
    hdf5_file = os.path.join(data_dir, 'hdf5_file.hdf5')
    input_group_name = TEST_EXPECTED_VOLUME_GROUPS[0]
    streamline_group_name = TEST_EXPECTED_STREAMLINE_GROUPS[0]

    # Here, testing default values only. See dwi_ml.tests.test_trainer for more
    # various testing.
    # Max length in current testing dataset is 108. Setting max length to 115
    # for faster testing.
    logging.info("************ TESTING TRAINING ************")
    ret = script_runner.run('tt_train_original_model.py',
                            experiments_path, experiment_name, hdf5_file,
                            input_group_name, streamline_group_name,
                            '--max_epochs', '1', '--batch_size', '5',
                            '--batch_size_units', 'nb_streamlines',
                            '--max_batches_per_epoch', '5',
                            '--max_len', '115',
                            '--logging', 'INFO')
    assert ret.success
    #
    # logging.info("************ TESTING RESUMING FROM CHECKPOINT ************")
    # ret = script_runner.run('l2t_resume_training_from_checkpoint.py',
    #                         experiment_path, 'test_experiment',
    #                         '--new_max_epochs', '2')
    # assert ret.success
    #
    # logging.info("************ TESTING TRACKING FROM MODEL ************")
    # whole_experiment_path = os.path.join(experiment_path, experiment_name)
    # out_tractogram = os.path.join(tmp_dir.name, 'test_tractogram.trk')
    # ret = script_runner.run(
    #     'l2t_track_from_model.py', whole_experiment_path, out_tractogram,
    #     'det', '--nt', '2', '--logging', 'debug',
    #     '--sm_from_hdf5', TEST_EXPECTED_VOLUME_GROUPS[1],
    #     '--tm_from_hdf5', TEST_EXPECTED_VOLUME_GROUPS[1],
    #     '--input_from_hdf5', TEST_EXPECTED_VOLUME_GROUPS[0],
    #     '--hdf5_file', hdf5_file, '--subj_id', TEST_EXPECTED_SUBJ_NAMES[0])
    #
    # assert ret.success
