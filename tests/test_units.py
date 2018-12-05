"""Unit testing."""
import pytest
import numpy as np

import convising.train as tr


def test_cg_deci():

    config = tr.Config()
    config.L = 4
    config.cg_method = "deci"
    config.cg_factor = 2

    test_images = np.array([[[ 1,  1,  1,  1],
                             [ 1,  1,  1,  1],
                             [ 1,  1,  1,  1],
                             [ 1,  1,  1,  1]],
                            [[-1,  1,  1,  1],
                             [ 1,  1,  1,  1],
                             [-1, -1,  1, -1],
                             [-1, -1, -1, -1]],
                            [[-1, -1, -1, -1],
                             [-1, -1, -1, -1],
                             [-1, -1, -1, -1],
                             [-1, -1, -1, -1]]])
    ([test_cg, _], _) = tr.coarse_grain(
        config.L,
        config.beta,
        config.cg_method,
        config.cg_factor,
        test_images)
    deci_test_images = np.array([[[ 1,  1],
                                  [ 1,  1]],
                                 [[-1,  1],
                                  [-1,  1]],
                                 [[-1, -1],
                                  [-1, -1]]])

    assert np.array_equal(deci_test_images, test_cg)

def test_cg_maj():

    config = tr.Config()
    config.L = 4
    config.cg_method = "maj"
    config.cg_factor = 2

    test_images = np.array([[[ 1,  1,  1,  1],
                             [ 1,  1,  1,  1],
                             [ 1,  1,  1,  1],
                             [ 1,  1,  1,  1]],
                            [[-1,  1,  1,  1],
                             [ 1,  1,  1,  1],
                             [-1, -1,  1, -1],
                             [-1, -1, -1, -1]],
                            [[-1, -1, -1, -1],
                             [-1, -1, -1, -1],
                             [-1, -1, -1, -1],
                             [-1, -1, -1, -1]]])
    ([test_cg, _], _) = tr.coarse_grain(
        config.L,
        config.beta,
        config.cg_method,
        config.cg_factor,
        test_images)
    maj_test_images = np.array([[[ 1,  1],
                                 [ 1,  1]],
                                [[ 1,  1],
                                 [-1, -1]],
                                [[-1, -1],
                                 [-1, -1]]])

    assert np.array_equal(maj_test_images, test_cg)
