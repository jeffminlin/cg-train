import os

import h5py
import numpy as np


def load_datasets(
    datafile,
    config_ising,
    config_train,
    cg_level_start=1,
    cg_level_end=2,
    shuffle=True,
    cg_ref_file=None,
):

    cg_method = config_ising["cg_method"]
    cg_factor = config_ising["cg_factor"]

    for cg_level in range(cg_level_start, cg_level_end + 1):
        outdata = {}
        labels = {}

        with h5py.File(datafile, "r") as dset:
            if cg_level == 0:
                images = dset["images"]
                imagesets, exp_ediffs = coarse_grain(
                    config_ising["L"],
                    config_ising["beta"],
                    cg_method,
                    cg_factor,
                    images,
                )
                images_flipped = imagesets[1]
            else:
                group = str(cg_level) + "/" + cg_method + "/" + str(cg_factor)
                images = dset[group]["images"]
                images_flipped = dset[group]["images_flipped"]
                exp_ediffs = dset[group]["exp_ediffs"]

            num_samples = images.len()
            idx = np.arange(num_samples)
            if shuffle:
                np.random.shuffle(idx)
            imagedata = split_train_val_test(images, config_train, idx)
            imagedata_flipped = split_train_val_test(images_flipped, config_train, idx)
        # Concatentate
        for key in imagedata:
            outdata[cg_level][key] = [imagedata, imagedata_flipped]

        labels[cg_level] = split_train_val_test(exp_ediffs, config_train, idx)

    if cg_ref_file:
        with h5py.File(datafile, "r") as dset:
            cg_exp_ediffs = compute_exact_cg(config_ising, dset, cg_ref_file)
            exact_labels = split_train_val_test(cg_exp_ediffs, config_train, idx)

        return outdata, labels, exact_labels

    return outdata, labels


def split_train_val_test(h5data, config_train, idx):

    train_split = config_train["train_split"]
    val_split = config_train["val_split"]

    num_samples = h5data.len()
    num_train = int(num_samples * (train_split - train_split * val_split))
    num_val = int(num_samples * train_split * val_split)
    num_test = num_samples - num_train - num_val

    idx_train = sorted(idx[:num_train])
    idx_val = sorted(idx[num_train : num_train + num_val])
    idx_test = sorted(idx[-num_test:])

    if num_val:
        return {
            "train": h5data[idx_train],
            "val": h5data[idx_val],
            "test": h5data[idx_test],
        }

    return {"train": h5data[idx_train], "test": h5data[idx_test]}


def create_cg_dataset(config_ising, datafile, cg_level_start, cg_level_end):

    L = config_ising["L"]
    beta = config_ising["beta"]
    cg_method = config_ising["cg_method"]
    cg_factor = config_ising["cg_factor"]

    with h5py.File(datafile, "r+") as dset:
        if cg_level_start == 0:
            imagearray = dset["images"][:, :]
        else:
            cgpath = str(cg_level_start) + "/" + cg_method + "/" + str(cg_factor) + "/"
            imagearray = dset[cgpath + "images"][:, :]

        for flipidx in range(cg_level_end - cg_level_start):
            cgL = int(L / (cg_factor)**(cg_level_start + flipidx))
            print("Coarse graining from", cgL , "to", int(cgL / cg_factor))
            cg_images, exp_ediff = coarse_grain(
                cgL, beta, cg_method, cg_factor, imagearray
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
                    dset, ["1", cg_method, str(cg_factor)], "exp_ediffs", data=exp_ediff
                )
            imagearray = cg_images[0]


def compute_exact_cg(config_ising, dset, cg_ref_file):

    L = config_ising["L"]
    beta = config_ising["beta"]
    cg_method = config_ising["cg_method"]
    cg_factor = config_ising["cg_factor"]

    cg_path = "1/" + cg_method + "/" + cg_factor + "/"
    cgimage = dset[cg_path + "images"]
    cgimageflip = dset[cg_path + "images_flipped"]

    with h5py.File(cg_ref_file, "r") as cgref_dset:
        cgref_ss = cgref_dset["images"][:, :]
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

    try:
        group = dset.create_group("/".join(grouplist))
        print("Group", "/".join(grouplist), "created")
    except ValueError as error:
        print("Group", "/".join(grouplist), "already exists")
        group = dset["/".join(grouplist)]

    try:
        group.create_dataset(name, data=data, dtype=dtype)
        print("Coarse-grained data created,", name)
    except RuntimeError as error:
        print("Coarse-grained data likely already created,", name + ",", "overwriting")
        del group[name]
        group.create_dataset(name, data=data, dtype=dtype)
