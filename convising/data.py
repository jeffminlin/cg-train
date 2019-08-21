import os

import h5py
import numpy as np


def create_cg_dataset(
    L, beta, cg_method, cg_factor, datapath, cg_level_start, cg_level_end
):

    datafile = os.path.join(datapath, "L{0:d}b{1:.4e}.h5".format(L, beta))
    with h5py.File(datafile, "r+") as dset:
        if cg_level_start == 0:
            imagearray = dset["images"][:, :]
        else:
            cgpath = str(cg_level_start) + "/" + cg_method + "/" + cg_factor + "/"
            imagearray = dset[cgpath + "images"][:, :]

        for flipidx in range(cg_level_end - cg_level_end - 1):
            cg_images, exp_ediff = coarse_grain(
                L, beta, cg_method, cg_factor, imagearray
            )
            h5py_create_data_catch(
                dset,
                [str(cg_level_start + flipidx + 1), cg_method, str(cg_factor)],
                "images",
                data=cg_images[0],
                dtype="i1",
            )
            h5py_create_data_catch(
                dset,
                [str(cg_level_start + flipidx + 1), cg_method, str(cg_factor)],
                "images_flipped",
                data=cg_images[1],
                dtype="i1",
            )
            if cg_level_start == 0 and flipidx == 0:
                h5py_create_data_catch(
                    dset, ["1", cg_method, str(cg_factor)], "exp_ediff", data=exp_ediff
                )


def compute_exact_cg(L, beta, cg_method, cg_factor, dset, cg_level, cg_ref_file):

    cg_path = str(cg_level) + "/" + cg_method + "/" + cg_factor + "/"
    cgimage = dset[cg_path + "images"]
    cgimageflip = dset[cg_path + "images_flipped"]

    cgref_dset = h5py.File(cg_ref_file, "r")
    cgref_ss = cgref_dset["images"]
    cgref_e = cgref_dset["energies"][:] / beta

    cg_e = lookup_cg_e(L, beta, cg_factor, cgimage, cgref_ss, cgref_e)
    cg_e_flip = lookup_cg_e(L, beta, cg_factor, cgimageflip, cgref_ss, cgref_e)
    cg_exp_ediff = np.exp(-beta * (cg_e_flip - cg_e))

    return cg_exp_ediff


def coarse_grain(L, beta, cg_method, cgf, image):

    image = image.reshape([-1, L, L])
    numdat = len(image)
    cgL = int(L / cgf)
    cgflipidx = np.zeros((2, numdat), dtype=int)
    # cgflipidx = np.random.randint(cgL, size=(2, numdat))
    flipidx = cgflipidx * cgf
    exp_ediff = cg_ediff(L, beta, cg_method, cgf, image, flipidx)
    cgimage, cgimageflip = cg_imageflip(L, cg_method, cgf, image, cgflipidx)

    return ([cgimage, cgimageflip], exp_ediff)


def shift_flipidx(L, flipidx, shiftarr):

    numdat = flipidx.shape[1]
    outidx = np.insert((flipidx + shiftarr) % [[L], [L]], 0, range(numdat), axis=0)

    return tuple(outidx)


def cg_ediff(L, beta, cg_method, cgf, image, flipidx):

    numdat = len(image)
    if cg_method == "deci":
        addidx = [[[1], [0]], [[0], [1]], [[-1], [0]], [[0], [-1]]]
        ediff = 2 * (
            image[tuple(np.insert(flipidx, 0, range(numdat), axis=0))]
            * np.sum(
                [image[shift_flipidx(L, flipidx, shift)] for shift in addidx], axis=0
            )
        )
    elif cg_method == "maj":
        ediff = 2 * np.sum(
            [
                image[shift_flipidx(L, flipidx, np.array([[blkshift], [shift]]))]
                * image[shift_flipidx(L, flipidx, np.array([[blkshift - 1], [shift]]))]
                + image[shift_flipidx(L, flipidx, np.array([[shift], [blkshift]]))]
                * image[shift_flipidx(L, flipidx, np.array([[shift - 1], [blkshift]]))]
                for blkshift in [0, cgf]
                for shift in range(cgf)
            ],
            axis=0,
        )

    exp_ediff = np.exp(-beta * ediff)

    return exp_ediff


def cg_imageflip(L, cg_method, cgf, image, cgflipidx):

    numdat = len(image)
    # Decimate
    cgimage = image[:, 0::cgf, 0::cgf]

    # If majority rule, replace cgimage elements with the sign of the sum of blocks
    if cg_method == "maj":
        for cgidx, _ in np.ndenumerate(cgimage[0]):
            cgblock = image[
                :,
                (cgidx[0] * cgf) : ((cgidx[0] + 1) * cgf),
                (cgidx[1] * cgf) : ((cgidx[1] + 1) * cgf),
            ]
            cgimage[(slice(None), *cgidx)] = np.sign(np.sum(cgblock, axis=(1, 2)))
        ss_zero_idx = cgimage == 0
        nnz = np.count_nonzero(ss_zero_idx)
        rand_ss = np.random.choice([-1, 1], nnz)
        cgimage[ss_zero_idx] = rand_ss

    # Flip the given cg indices
    cgimageflip = np.copy(cgimage)
    cgimageflip[tuple(np.insert(cgflipidx, 0, range(numdat), axis=0))] *= -1

    return cgimage, cgimageflip


def lookup_cg_e(L, beta, cgf, cgimage, cgref_ss, cgref_e):

    im_equal = [
        np.argwhere(np.sum(np.square(cgref_ss - image.ravel()), axis=1) == 0)
        for image in cgimage
    ]
    cg_e = np.array([cgref_e[idx] for idx in im_equal]).ravel()

    return cg_e


def h5py_create_data_catch(dset, grouplist, name, data, dtype=None):

    if dtype is None:
        dtype = data.dtype

    group = dset["/".join(grouplist)]

    try:
        group.create_dataset(name, data=data, dtype=dtype)
        print("Coarse-grained data created,", name)
    except RuntimeError as error:
        print(error)
        print("Coarse-grained data likely already created,", name)
