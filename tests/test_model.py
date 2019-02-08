"""Testing learned model symmetries."""
import pytest
import numpy as np
from numpy.testing import assert_allclose

import convising.train as tr


def test_z2_sym_L8_log_cosh():

    config = tr.Config()
    config.L = 8
    config.cg_method = 'deci'
    config.conv_activ = 'log_cosh'
    config.refresh_config()
    deep_conv = tr.ConvIsing(config)
    deep_conv.reload_weights(config)

    test_input = np.random.choice([-1, 1], size=(50, config.cgL, config.cgL))
    test_out = deep_conv.model_energy.predict(test_input)
    test_out_flip = deep_conv.model_energy.predict(-test_input)

    assert_allclose(test_out, test_out_flip, rtol=1.0e-6)


def test_translate_sym_L8():

    config = tr.Config()
    config.L = 8
    config.cg_method = 'deci'
    config.refresh_config()
    deep_conv = tr.ConvIsing(config)
    deep_conv.reload_weights(config)

    test_input = np.random.choice([-1, 1], size=(50, config.cgL, config.cgL))
    test_input_roll = np.roll(test_input, np.random.randint(config.cgL, size=2), axis=(1, 2))
    test_out = deep_conv.model_energy.predict(test_input)
    test_out_roll = deep_conv.model_energy.predict(test_input_roll)

    assert_allclose(test_out, test_out_roll, rtol=1.0e-4)


def test_translate_sym_L8_log_cosh():

    config = tr.Config()
    config.L = 8
    config.cg_method = 'deci'
    config.conv_activ = 'log_cosh'
    config.refresh_config()
    deep_conv = tr.ConvIsing(config)
    deep_conv.reload_weights(config)

    test_input = np.random.choice([-1, 1], size=(50, config.cgL, config.cgL))
    test_input_roll = np.roll(test_input, np.random.randint(config.cgL, size=2), axis=(1, 2))
    test_out = deep_conv.model_energy.predict(test_input)
    test_out_roll = deep_conv.model_energy.predict(test_input_roll)

    assert_allclose(test_out, test_out_roll, rtol=1.0e-4)
