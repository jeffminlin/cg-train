"""Testing RG requirements."""
import pytest
import numpy as np

import convising.data as data


def compute_lattice_sum(image, block_kernel):

    lattice_sum = 0
    for cg_config in range(2 ** 4):
        cgimage = np.array([int(spin) for spin in np.binary_repr(cg_config).zfill(4)])
        ss_zero_idx = cgimage == 0
        cgimage[ss_zero_idx] = -1
        lattice_sum += data.lattice_kernel(image, cgimage.reshape((2, 2)), block_kernel)

    return lattice_sum


def test_deci_lattice_sum():

    image = np.array([[1, -1, 1, 1], [-1, 1, -1, 1], [1, 1, -1, -1], [1, 1, -1, -1]])
    lattice_sum = compute_lattice_sum(image, data.block_deci_kernel)

    assert lattice_sum == 1.0


def test_maj_lattice_sum():

    image = np.array([[1, -1, 1, 1], [-1, 1, -1, 1], [1, 1, -1, -1], [1, 1, -1, -1]])
    lattice_sum = compute_lattice_sum(image, data.block_maj_kernel)

    assert lattice_sum == 1.0
