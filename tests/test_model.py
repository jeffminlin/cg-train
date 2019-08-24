"""Testing model symmetries."""
import pytest
import numpy as np
from numpy.testing import assert_allclose

import convising.train as train
import convising.models as models


def test_z2_sym_log_cosh():

    config_ising = {"beta": 0.5}
    config_train = {
        "conv_activation": "log_cosh",
        "nfilters": 2,
        "kernel_size": 1,
        "dense_nodes": [2],
        "dense_activation": "relu",
    }

    model_group = models.ModelGroup(config_ising, config_train)

    test_input = np.random.choice([-1, 1], size=(50, 4, 4))
    test_out = model_group.energy.predict(test_input)
    test_out_flip = model_group.energy.predict(-test_input)

    assert_allclose(test_out, test_out_flip, rtol=1.0e-6)


def test_translate_sym_L8():

    config_ising = {"beta": 0.5}
    config_train = {
        "conv_activation": "linear",
        "nfilters": 2,
        "kernel_size": 1,
        "dense_nodes": [2],
        "dense_activation": "relu",
    }

    model_group = models.ModelGroup(config_ising, config_train)

    test_input = np.random.choice([-1, 1], size=(50, 4, 4))
    test_input_roll = np.roll(test_input, np.random.randint(4, size=2), axis=(1, 2))
    test_out = model_group.energy.predict(test_input)
    test_out_roll = model_group.energy.predict(test_input_roll)

    assert_allclose(test_out, test_out_roll, rtol=1.0e-4)


def test_translate_sym_log_cosh():

    config_ising = {"beta": 0.5}
    config_train = {
        "conv_activation": "log_cosh",
        "nfilters": 2,
        "kernel_size": 1,
        "dense_nodes": [2],
        "dense_activation": "relu",
    }

    model_group = models.ModelGroup(config_ising, config_train)

    test_input = np.random.choice([-1, 1], size=(50, 4, 4))
    test_input_roll = np.roll(test_input, np.random.randint(4, size=2), axis=(1, 2))
    test_out = model_group.energy.predict(test_input)
    test_out_roll = model_group.energy.predict(test_input_roll)

    assert_allclose(test_out, test_out_roll, rtol=1.0e-4)
