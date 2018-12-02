import pytest
import numpy as np

import convising.train as tr


def test_cg_deci():

    config = tr.Config()
    config.L = 4
    config.cg_method = "deci"
    config.cg_factor = 2

    testimages = np.random.choice([-1,1], size=(3,4,4))

    ([testcg,testcgflip],testediff) = tr.coarse_grain(config.L, config.beta, config.cg_method, config.cg_factor, testimages)

    assert np.array_equal(testimages[:,0::config.cg_factor,0::config.cg_factor], testcg)
