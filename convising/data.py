import h5py
import numpy as np


def create_cg_dataset(config_ising, config_train, start_cg, end_cg):
    with h5py.File(config_train["datafile"], "r+") as dset:
        if start_cg == 0:
            imagearray = dset["images"][:, :]
        else:
            imagearray = dset[
                "".join(
                    [
                        "cgimage",
                        str(start_cg),
                        "_",
                        config_ising["cg_method"],
                        str(config_ising["cg_factor"]),
                    ]
                )
            ]

        for flipidx in range(end_cg - start_cg - 1):
            cg_images, exp_ediff = coarse_grain(
                config_ising["L"],
                config_ising["beta"],
                config_ising["cg_method"],
                config_ising["cgf"],
                imagearray,
            )
            cgimage = cg_images[0]
            cgimageflip = cg_images[1]

            h5py_create_data_catch(
                dset,
                "".join(
                    [
                        "cgimage",
                        str(start_cg + flipidx + 1),
                        "_",
                        config_ising["cg_method"],
                        str(config_ising["cg_factor"]),
                    ]
                ),
                data=cgimage,
                dtype="i1",
            )
            h5py_create_data_catch(
                dset,
                "".join(
                    [
                        "cgimageflip",
                        str(start_cg + flipidx + 1),
                        "_",
                        config_ising["cg_method"],
                        str(config_ising["cg_factor"]),
                    ]
                ),
                data=cgimageflip,
                dtype="i1",
            )
            if start_cg == 0 and flipidx == 0:
                h5py_create_data_catch(
                    dset,
                    "".join(
                        [
                            "exp_ediff1_",
                            config_ising["cg_method"],
                            str(config_ising["cg_factor"]),
                        ]
                    ),
                    data=exp_ediff,
                )

            imagearray = cgimage


def compute_exact_cg(dset, cg_ref_file, config_ising):
    cgimage = dset[
        "".join(["cgimage_", config_ising["cg_method"], str(config_ising["cg_factor"])])
    ]
    cgimageflip = dset[
        "".join(
            ["cgimageflip_", config_ising["cg_method"], str(config_ising["cg_factor"])]
        )
    ]
    cgref_dset = h5py.File(cg_ref_file, "r")
    cgref_ss = cgref_dset["images"]
    cgref_e = cgref_dset["energies"][:] / config_ising["beta"]

    cg_e = lookup_cg_e(
        config_ising["L"],
        config_ising["beta"],
        config_ising["cg_factor"],
        cgimage,
        cgref_ss,
        cgref_e,
    )
    cg_e_flip = lookup_cg_e(
        config_ising["L"],
        config_ising["beta"],
        config_ising["cg_factor"],
        cgimageflip,
        cgref_ss,
        cgref_e,
    )
    cg_exp_ediff = np.exp(-config_ising["beta"] * (cg_e_flip - cg_e))


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


def cg_ediff(L, beta, cg_method, cgf, image, flipidx):

    numdat = len(image)
    if cg_method == "deci":
        addidx = [[[1], [0]], [[0], [1]], [[-1], [0]], [[0], [-1]]]
        ediff = 2 * (
            image[tuple(np.insert(flipidx, 0, range(numdat), axis=0))]
            * np.sum(
                [
                    image[
                        tuple(
                            np.insert(
                                (flipidx + shift) % [[L], [L]], 0, range(numdat), axis=0
                            )
                        )
                    ]
                    for shift in addidx
                ],
                axis=0,
            )
        )
    elif cg_method == "maj":
        ediff = 2 * np.sum(
            [
                image[
                    tuple(
                        np.insert(
                            (flipidx + np.array([[blkshift], [shift]])) % [[L], [L]],
                            0,
                            range(numdat),
                            axis=0,
                        )
                    )
                ]
                * image[
                    tuple(
                        np.insert(
                            (flipidx + np.array([[blkshift - 1], [shift]]))
                            % [[L], [L]],
                            0,
                            range(numdat),
                            axis=0,
                        )
                    )
                ]
                + image[
                    tuple(
                        np.insert(
                            (flipidx + np.array([[shift], [blkshift]])) % [[L], [L]],
                            0,
                            range(numdat),
                            axis=0,
                        )
                    )
                ]
                * image[
                    tuple(
                        np.insert(
                            (flipidx + np.array([[shift - 1], [blkshift]]))
                            % [[L], [L]],
                            0,
                            range(numdat),
                            axis=0,
                        )
                    )
                ]
                for blkshift in [0, cgf]
                for shift in range(cgf)
            ],
            axis=0,
        )

    exp_ediff = np.exp(-beta * ediff)

    return exp_ediff


def cg_imageflip(L, cg_method, cgf, image, cgflipidx):

    numdat = len(image)
    cgimage = image[:, 0::cgf, 0::cgf]
    if cg_method == "maj":
        for cgidx, _ in np.ndenumerate(cgimage[0]):
            cgimage[(slice(None), *cgidx)] = np.sign(
                np.sum(
                    image[
                        :,
                        (cgidx[0] * cgf) : ((cgidx[0] + 1) * cgf),
                        (cgidx[1] * cgf) : ((cgidx[1] + 1) * cgf),
                    ],
                    axis=(1, 2),
                )
            )
        ss_zero_idx = cgimage == 0
        nnz = np.count_nonzero(ss_zero_idx)
        rand_ss = np.random.choice([-1, 1], nnz)
        cgimage[ss_zero_idx] = rand_ss

    cgimageflip = np.copy(cgimage)
    cgimageflip[tuple(np.insert(cgflipidx, 0, range(numdat), axis=0))] = cgimageflip[
        tuple(np.insert(cgflipidx, 0, range(numdat), axis=0))
    ] * (-1)

    return cgimage, cgimageflip


def lookup_cg_e(L, beta, cgf, cgimage, cgref_ss, cgref_e):

    im_equal = [
        np.argwhere(np.sum(np.square(cgref_ss - image.ravel()), axis=1) == 0)
        for image in cgimage
    ]
    cg_e = np.array([cgref_e[idx] for idx in im_equal]).ravel()

    return cg_e


class GeneratorIsing(tf.keras.utils.Sequence):
    """Class for feeding data to the fit_generator method of a Keras model.

    Args:
        dset
            h5 data with "cgimages" and "ediff" datasets containing
            coarse-grained Ising configurations
        config (Config)
            instance of Config class
        set (str)
            whether to give training ("train"), validation ("val"),
            or testing ("test") samples

    """

    def __init__(self, dset, config, num_cg, shuffle=True, set="test", flip=True):
        self.exp_ediff = dset[
            "".join(["exp_ediff1_", config.cg_method, str(config.cg_factor)])
        ]
        numdat = int(len(self.exp_ediff) * config.keep)
        num_train = int(numdat * config.train_split * (1 - config.val_split))
        num_val = int(numdat * config.train_split * config.val_split)
        num_test = numdat - num_train - num_val
        if set == "train":
            self.idxset = np.arange(num_train)
        elif set == "val":
            self.idxset = np.arange(num_val) + num_train
        else:
            self.idxset = np.arange(num_test) + num_train + num_val
        self.num_samples = len(self.idxset)

        self.image = dset[
            "".join(
                ["cgimage", str(num_cg), "_", config.cg_method, str(config.cg_factor)]
            )
        ][self.idxset, :, :]
        self.imageflip = dset[
            "".join(
                [
                    "cgimageflip",
                    str(num_cg),
                    "_",
                    config.cg_method,
                    str(config.cg_factor),
                ]
            )
        ][self.idxset, :, :]
        self.exp_ediff = self.exp_ediff[self.idxset.tolist()]

        self.flip = flip
        self.indexes = np.arange(self.num_samples)
        self.batch_size = config.batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.num_samples / float(self.batch_size)))

    def __getitem__(self, idx):
        # Generate unprocessed batch
        batch_image = self.image[
            self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size].tolist(),
            :,
            :,
        ]
        batch_imageflip = self.imageflip[
            self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size].tolist(),
            :,
            :,
        ]
        batch_ediff = self.exp_ediff[
            self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size].tolist()
        ]

        if self.flip:
            return ([batch_image, batch_imageflip], batch_ediff)
        else:
            return ([batch_image], batch_ediff)

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


def h5py_create_data_catch(dset, dataname, data, dtype=None):

    if dtype is None:
        dtype = data.dtype

    try:
        dset.create_dataset(dataname, data=data, dtype=dtype)
        print("Coarse-grained data created,", dataname)
    except RuntimeError as error:
        print(error)
        print("Coarse-grained data likely already created,", dataname)
