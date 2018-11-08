import numpy as np


class IsingCG(Sequence):

    def __init__(self, config, set="test"):
        self.L = config.L
        self.beta = config.beta
        self.imagearray = np.load(config.imagefile, mmap_mode="r")
        self.earray = np.load(config.efile, mmap_mode="r")
        num_samples = len(self.earray)
        num_train = int(num_samples*(config.train_split - config.train_split*config.val_split))
        num_val = int(num_samples*config.train_split*config.val_split)
        num_test = num_samples - num_train - num_val
        if set == "train":
            self.images = self.imagearray[:num_train]
            self.energies = self.earray[:num_train]
        elif set == "val":
            self.images = self.imagearray[num_train:num_train+num_val]
            self.energies = self.earray[num_train:num_train+num_val]
        else:
            self.images = self.imagearray[-num_test:]
            self.energies = self.earray[-num_test:]
        self.num_samples = len(self.energies)

        self.cg_method = config.cg_method
        self.cgf = config.cg_factor
        self.cgL = int(self.L/self.cgf)

        self.batch_size = config.batch_size
        self.dim = config.dim
        self.num_channels = config.num_channels
        self.shuffle = config.shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.num_samples / float(self.batch_size)))

    def __getitem__(self, idx):
        # Generate indexes of the batch
        batch_image = self.images[self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size],:]
        batch_energy = self.energies[self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]]

        batch_imagepair, batch_ediff = coarse_grain(batch_image, batch_energy)

        return batch_imagepair, batch_ediff

    def on_epoch_end(self):
        self.indexes = np.arange(self.num_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


def coarse_grain(L, beta, cg_method, cgf, image):

    numdat = len(image)
    cgL = int(L/cgf)
    image = image.reshape([-1,L,L])
    cgflipidx = np.random.randint(cgL, size=(2, numdat))
    flipidx = cgflipidx*cgf
    exp_ediff = cg_ediff(L, beta, cg_method, cgf, image, flipidx)
    cgimage,cgimageflip = cg_imageflip(L, cg_method, cgf, image, cgflipidx)

    return ([cgimage, cgimageflip], exp_ediff)


def cg_ediff(L, beta, cg_method, cgf, image, flipidx):

    numdat = len(image)
    if cg_method == "deci":
        addidx = [[[1],[0]], [[0],[1]], [[-1],[0]], [[0],[-1]]]
        ediff = 2 * (image[np.insert(flipidx, 0, range(numdat), axis=0).tolist()] *
            np.sum([image[np.insert((flipidx + shift)%[[L],[L]], 0, range(numdat), axis=0).tolist()] for shift in addidx], axis=0))
    elif cg_method == "maj":
        ediff = 2 * np.sum(
            [image[np.insert((flipidx + [[blkshift],[shift]])%[[L],[L]], 0, range(numdat), axis=0).tolist()] * image[np.insert(flipidx + [[blkshift-1],[shift]]%[[L],[L]], 0, range(numdat), axis=0).tolist()]
            + image[np.insert((flipidx + [[shift],[blkshift]])%[[L],[L]], 0, range(numdat), axis=0).tolist()] * image[np.insert((flipidx + [[shift-1],[blkshift]])%[[L],[L]], 0, range(numdat), axis=0).tolist()] for blkshift in [0,cgf] for shift in range(cgf)], axis=0)

    exp_ediff = np.exp(-beta*ediff)

    return exp_ediff


def cg_imageflip(L, cg_method, cgf, image, cgflipidx):

    numdat = len(image)
    cgimage = image[:,0::cgf,0::cgf]
    if cg_method == "maj":
        for cgidx,_ in np.ndenumerate(cgimage[0]):
            cgimage[(slice(None),*cgidx)] = np.sign(np.sum(image[:,cgidx[0]*cgf:(cgidx[0] + 1)*cgf,cgidx[1]*cgf:(cgidx[1] + 1)*cgf], axis=(1,2)))
        ss_zero_idx = (cgimage == 0)
        nnz = np.count_nonzero(ss_zero_idx)
        rand_ss = np.random.choice([-1,1], nnz)
        cgimage[ss_zero_idx] = rand_ss

    cgimageflip = np.copy(cgimage)
    cgimageflip[np.insert(cgflipidx, 0, range(numdat), axis=0).tolist()] = cgimageflip[np.insert(cgflipidx, 0, range(numdat), axis=0).tolist()]*(-1)

    return cgimage, cgimageflip
