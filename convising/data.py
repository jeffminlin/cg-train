import os
import time

import h5py
import numpy as np


def load_datasets(
    datafile,
    config_ising,
    config_train,
    cg_level_start=1,
    cg_level_end=2,
    shuffle=True,
    exact=False,
):

    cg_method = config_ising["cg_method"]
    cg_factor = config_ising["cg_factor"]
    outdata = {}
    labels = {}

    for cg_level in range(cg_level_start, cg_level_end + 1):
        outdata[cg_level] = {}

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
                images = dset[group]["images"][:, :, :]
                images_flipped = dset[group]["images_flipped"][:, :, :]

            num_samples = int(len(images) * config_train["keep_data"])
            idx = np.arange(num_samples)
            if shuffle:
                np.random.shuffle(idx)
            imagedata = split_train_val_test(images, config_train, idx)
            imagedata_flipped = split_train_val_test(images_flipped, config_train, idx)

            if cg_level == 1:
                exp_ediffs = dset[group]["exp_ediffs"][:]
                labels = split_train_val_test(exp_ediffs, config_train, idx)
                if exact:
                    exp_cg_ediffs = dset[group]["exp_cg_ediffs"][:]
                    exact_labels = split_train_val_test(
                        exp_cg_ediffs, config_train, idx
                    )

        for key in imagedata:  # Concatentate data
            outdata[cg_level][key] = [
                imagedata[key].astype(np.float64),
                imagedata_flipped[key].astype(np.float64),
            ]

    if exact:
        return outdata, labels, exact_labels

    return outdata, labels


def split_train_val_test(h5data, config_train, idx):

    datasets = {}

    train_split = config_train["train_split"]
    val_split = config_train["val_split"]

    num_samples = len(idx)
    num_train = int(num_samples * (train_split - train_split * val_split))
    num_val = int(num_samples * train_split * val_split)
    num_test = num_samples - num_train - num_val

    idx_train = sorted(idx[:num_train])
    datasets["train"] = h5data[idx_train]
    if num_val:
        idx_val = sorted(idx[num_train : num_train + num_val])
        datasets["val"] = h5data[idx_val]
    idx_test = sorted(idx[-num_test:])
    datasets["test"] = h5data[idx_test]

    return datasets


def create_cg_dataset(
    config_ising,
    datafile,
    cg_level_start,
    cg_level_end,
    overwrite=False,
    cg_ref_file=None,
):

    L = config_ising["L"]
    beta = config_ising["beta"]
    cg_method = config_ising["cg_method"]
    cg_factor = config_ising["cg_factor"]

    with h5py.File(datafile, "r+") as dset:
        if cg_level_start == 0:
            imagearray = dset["images"][:, :]
        else:
            cgpath = "/".join([str(cg_level_start), cg_method, str(cg_factor)])
            imagearray = dset[cgpath]["images"][:, :]

        for flipidx in range(cg_level_end - cg_level_start):
            curlevel = cg_level_start + flipidx + 1
            cgL = int(L / ((cg_factor) ** (cg_level_start + flipidx)))
            print("Coarse graining from", cgL, "to", int(cgL / cg_factor))

            grouplist = [str(curlevel), cg_method, str(cg_factor)]
            exists_check = check_group_datasets(dset, grouplist, curlevel, cg_ref_file)
            no_overwrite = exists_check and not overwrite

            if no_overwrite:
                print("Dataset exists, images")
                cgpath = "/".join([str(curlevel), cg_method, str(cg_factor)])
                imagearray = dset[cgpath]["images"][:, :, :]
            else:
                cg_images, exp_ediffs = coarse_grain(
                    cgL, beta, cg_method, cg_factor, imagearray
                )
                h5py_create_catch(
                    dset, grouplist, "images", data=cg_images[0], dtype="i1"
                )
                h5py_create_catch(
                    dset, grouplist, "images_flipped", data=cg_images[1], dtype="i1"
                )
                if curlevel == 1:
                    h5py_create_catch(
                        dset, grouplist, "exp_ediffs", data=exp_ediffs
                    )
                    if cg_ref_file:
                        exp_cg_ediffs = compute_exact_cg(
                            config_ising, dset, cg_ref_file
                        )
                        h5py_create_catch(
                            dset, grouplist, "exp_cg_ediffs", data=exp_cg_ediffs
                        )

                imagearray = cg_images[0]


def check_group_datasets(dset, grouplist, curlevel, cg_ref_file=None):

    exists_check = True
    images_exists = h5py_exists(dset, grouplist, "images")
    images_flipped_exists = h5py_exists(dset, grouplist, "images_flipped")
    exists_check = exists_check and images_exists and images_flipped_exists
    if curlevel == 1:
        exp_ediffs_exists = h5py_exists(dset, grouplist, "exp_ediffs")
        exists_check = exists_check and exp_ediffs_exists
        if cg_ref_file:
            exp_cg_ediffs_exists = h5py_exists(dset, grouplist, "exp_cg_ediffs")
            exists_check = exists_check and exp_cg_ediffs_exists

    return exists_check


def compute_exact_cg(config_ising, dset, cg_ref_file):

    L = config_ising["L"]
    beta = config_ising["beta"]
    cg_method = config_ising["cg_method"]
    cg_factor = config_ising["cg_factor"]

    cg_path = "1/" + cg_method + "/" + str(cg_factor) + "/"
    cgimage = dset[cg_path + "images"][:, :, :]
    cgimageflip = dset[cg_path + "images_flipped"][:, :, :]

    with h5py.File(cg_ref_file, "r") as cgref_dset:
        cgref_ss = cgref_dset["images"][:, :]
        cgref_e = cgref_dset["energies"][:] / beta

    cg_e = lookup_cg_e(L, beta, cg_factor, cgimage, cgref_ss, cgref_e)
    cg_e_flip = lookup_cg_e(L, beta, cg_factor, cgimageflip, cgref_ss, cgref_e)
    exp_cg_ediff = np.exp(-beta * (cg_e_flip - cg_e))

    return exp_cg_ediff


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

    cgref_dict = {}
    for idx, ss in enumerate(cgref_ss):
        cgref_dict[tuple(ss.ravel().tolist())] = cgref_e[idx]
    lookup_func = lambda image: cgref_dict[tuple(image.ravel().tolist())]
    cg_e = np.array([lookup_func(image) for image in cgimage]).ravel()

    return cg_e


def h5py_exists(dset, grouplist, name):

    return name in dset["/".join(grouplist)]


def h5py_create_catch(dset, grouplist, name, data, dtype=None, overwrite=True):

    if dtype is None:
        dtype = data.dtype

    try:
        group = dset.create_group("/".join(grouplist))
        print("Group", "/".join(grouplist), "created")
    except ValueError as error:
        group = dset["/".join(grouplist)]

    try:
        group.create_dataset(name, data=data, dtype=dtype)
        print("Dataset created,", name)
    except RuntimeError as error:
        print("A dataset is here,", name)
        if overwrite:
            print("Overwriting")
            del group[name]
            group.create_dataset(name, data=data, dtype=dtype)
