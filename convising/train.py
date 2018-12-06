import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras import optimizers, regularizers
from keras.engine.topology import Layer
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

import convising.models as mod


class Config:
    """Configuration for ConvIsing and MCMC


    Attributes:
        L (int): Size of original Ising lattice (LxL)
        beta (float): Inverse temperature
        cg_method (str): Type of coarse-graining (deci or maj)
        cg_factor (int): Coarse-graining factor

        keep (float): Fraction of data to keep for training and testing
        train_split (float): Fraction of kept data for training and validation
        val_split (float): Fraction of training data for validation
        batch_size (int): Number of data points per batch
        num_epochs (int): Maximum number of epochs to train
        verb (int): Verbosity parameter for keras fit
                (0 = no output, 1 = detailed output, 2 = some output)

        exact_cg (bool): Whether or not to use any data on exact cg values
        w_size (int): Filter window size, odd if nfsym = 'all' or 'z2'
        alpha (int): Number of filters
        conv_activ (str): Choice of activation function on the
            convolutional layer, either 'linear' or 'log_cosh'
        dense_nodes (int): List of integers describing the number of nodes in
                each dense layer
        nfsym (str): Symmetries to enforce, can be 'none', 'z2', 'd4', or 'all'

    """

    def __init__(self):
        self.L = 8
        self.beta = .4406868
        self.cg_method = "deci"
        self.cg_factor = 2

        self.keep = 1.0
        self.train_split = 9./10.
        self.val_split = 1./9.
        self.batch_size = 20000
        self.num_epochs = 1000
        self.verb = 0

        # Can set this to be True if L = 4 and the cg reference data is generated
        self.exact_cg = False

        self.w_size = 3
        self.alpha = 4
        self.conv_activ = 'linear'
        self.dense_nodes = [40, 40]
        self.nfsym = "none"

        self.refresh_config()

    def refresh_config(self):
        self.cgL = int(self.L/self.cg_factor)

        self.filepath = "".join(["L{0:d}b{1:.4e}", "a", str(self.alpha), "w", str(self.w_size), "conv", self.conv_activ, "_", *("".join(["n",str(dn)]) for dn in self.dense_nodes), self.cg_method, str(self.cg_factor), "nf", self.nfsym]).format(self.L, self.beta)
        self.weightfile = "".join(["./weights/", self.filepath, ".h5"])
        self.lossfile = "".join(["./figs/loss", self.filepath, ".png"])

        self.imagefile = "./data/ssL{0:d}b{1:.4e}.dat".format(self.L, self.beta)
        self.efile = "./data/EL{0:d}b{1:.4e}.dat".format(self.L, self.beta)

        self.cgref_imagefile = "".join(["./data/ssL{0:d}b{1:.4e}_cg", self.cg_method, str(self.cg_factor), ".dat"]).format(self.cgL, self.beta)
        self.cgref_efile = "".join(["./data/EL{0:d}b{1:.4e}_cg", self.cg_method, str(self.cg_factor), ".dat"]).format(self.cgL, self.beta)


def coarse_grain(L, beta, cg_method, cgf, image):

    numdat = len(image)
    cgL = int(L/cgf)
    image = image.reshape([-1,L,L])
    cgflipidx = np.zeros((2, numdat), dtype=int)
    # cgflipidx = np.random.randint(cgL, size=(2, numdat))
    flipidx = cgflipidx*cgf
    exp_ediff = cg_ediff(L, beta, cg_method, cgf, image, flipidx)
    cgimage, cgimageflip = cg_imageflip(L, cg_method, cgf, image, cgflipidx)

    return ([cgimage, cgimageflip], exp_ediff)


def cg_ediff(L, beta, cg_method, cgf, image, flipidx):

    numdat = len(image)
    if cg_method == "deci":
        addidx = [[[1],[0]], [[0],[1]], [[-1],[0]], [[0],[-1]]]
        ediff = 2 * (image[tuple(np.insert(flipidx, 0, range(numdat), axis=0))] *
            np.sum([image[tuple(np.insert((flipidx + shift)%[[L],[L]], 0, range(numdat), axis=0))] for shift in addidx], axis=0))
    elif cg_method == "maj":
        ediff = 2 * np.sum(
            [image[tuple(np.insert(
                (flipidx + np.array([[blkshift],[shift]]))%[[L],[L]],
                0, range(numdat), axis=0))]
            * image[tuple(np.insert(
                (flipidx + np.array([[blkshift-1],[shift]]))%[[L],[L]],
                0, range(numdat), axis=0))]
            + image[tuple(np.insert(
                (flipidx + np.array([[shift],[blkshift]]))%[[L],[L]],
                0, range(numdat), axis=0))]
            * image[tuple(np.insert(
                (flipidx + np.array([[shift-1],[blkshift]]))%[[L],[L]],
                0, range(numdat), axis=0))]
            for blkshift in [0,cgf] for shift in range(cgf)], axis=0)

    exp_ediff = np.exp(-beta*ediff)

    return exp_ediff


def cg_imageflip(L, cg_method, cgf, image, cgflipidx):

    numdat = len(image)
    cgimage = image[:,0::cgf,0::cgf]
    if cg_method == "maj":
        for cgidx,_ in np.ndenumerate(cgimage[0]):
            cgimage[(slice(None), *cgidx)] = np.sign(np.sum(
                image[:, (cgidx[0] * cgf):((cgidx[0] + 1) * cgf),
                (cgidx[1] * cgf):((cgidx[1] + 1) * cgf)], axis=(1,2)))
        ss_zero_idx = (cgimage == 0)
        nnz = np.count_nonzero(ss_zero_idx)
        rand_ss = np.random.choice([-1,1], nnz)
        cgimage[ss_zero_idx] = rand_ss

    cgimageflip = np.copy(cgimage)
    cgimageflip[tuple(np.insert(cgflipidx, 0, range(numdat), axis=0))] = cgimageflip[tuple(np.insert(cgflipidx, 0, range(numdat), axis=0))]*(-1)

    return cgimage, cgimageflip


def lookup_cg_e(L, beta, cgf, cgimage, cgref_ss, cgref_e):

    im_equal = [np.argwhere(np.sum(np.square(cgref_ss - image.ravel()), axis=1) == 0) for image in cgimage]
    cg_e = np.array([cgref_e[idx] for idx in im_equal]).ravel()

    return cg_e


class ConvIsing:
    """Class for training a neural network to learn the coarse-grained
        Hamiltonian for the Ising model.

    Args:
        config (Config): Configuration class

    """

    def __init__(self, config):
        self.L = config.L
        self.beta = config.beta
        self.cg_method = config.cg_method
        self.cgf = config.cg_factor
        self.verb = config.verb
        self.exact_cg = config.exact_cg
        self.create_model(config)

    def create_dataset(self, config):
        self.imagearray = np.loadtxt(config.imagefile)
        self.earray = np.loadtxt(config.efile)/self.beta
        self.numdat = int(len(self.earray)*config.keep)
        self.imagearray = self.imagearray[:self.numdat,:]
        self.earray = self.earray[:self.numdat]

        ([self.cgimage, self.cgimageflip], self.exp_ediff) = coarse_grain(self.L, self.beta, self.cg_method, self.cgf, self.imagearray)

        self.avg_e = np.mean(self.earray)
        self.avg_exp_e = np.mean(self.exp_ediff)

        self.imagearray = self.imagearray.reshape([-1,self.L,self.L])

        (self.train, self.val, self.test) = self.split_dataset(self.imagearray, config)
        (self.train_cg, self.val_cg, self.test_cg) = self.split_dataset(self.cgimage, config)
        (train_imageflip, val_imageflip, test_imageflip) = self.split_dataset(self.cgimageflip, config)
        self.train_traj = [self.train_cg, train_imageflip]
        self.val_traj = [self.val_cg, val_imageflip]
        self.test_traj = [self.test_cg, test_imageflip]

        (self.train_ediff, self.val_ediff, self.test_ediff) = self.split_dataset(self.exp_ediff, config)
        (self.train_e, self.val_e, self.test_e) = self.split_dataset(self.earray, config)

        if self.exact_cg:
            self.compute_exact_cg(config)

    def compute_exact_cg(self, config):
        self.cgref_ss = np.loadtxt(config.cgref_imagefile)
        self.cgref_e = np.loadtxt(config.cgref_efile)/self.beta

        self.cg_e = lookup_cg_e(self.L, self.beta, self.cgf, self.cgimage, self.cgref_ss, self.cgref_e)
        cg_e_flip = lookup_cg_e(self.L, self.beta, self.cgf, self.cgimageflip, self.cgref_ss, self.cgref_e)
        self.cg_exp_ediff = np.exp(-self.beta*(cg_e_flip - self.cg_e))
        self.avg_cg_e = np.mean(self.cg_e)
        self.avg_exp_cg_e = np.mean(self.cg_exp_ediff)

        (self.train_cg_ediff, self.val_cg_ediff, self.test_cg_ediff) = self.split_dataset(self.cg_exp_ediff, config)
        (self.train_cg_e, self.val_cg_e, self.test_cg_e) = self.split_dataset(self.cg_e, config)

    def split_dataset(self, data, config):
        num_samples = len(data)
        num_train = int(num_samples*(config.train_split - config.train_split*config.val_split))
        num_val = int(num_samples*config.train_split*config.val_split)
        num_test = num_samples - num_train - num_val

        train_data = data[:num_train]
        val_data = data[num_train:num_train+num_val]
        test_data = data[-num_test:]

        return (train_data, val_data, test_data)

    def create_model(self, config):
        K.clear_session()
        conv_activ = config.conv_activ
        activ_fcn = 'elu'
        kninit = 'glorot_normal'
        self.model_energy = mod.deep_conv_e(config, conv_activ, activ_fcn, kninit)
        self.model = mod.model_e_diff(config, self.model_energy)

    def run_model(self, config):
        self.model.compile(loss="mse", optimizer="Nadam")

        best_weight = ModelCheckpoint(config.weightfile, monitor="val_loss", save_best_only=True, save_weights_only=True)
        early_stop = EarlyStopping(monitor="val_loss", patience=100, mode='min') # Setting patience=config.num_epochs means the training does not stop early but it restores the best results (requires Keras 2.2.3, unsure if this is working)
        callback_list = [best_weight,early_stop]
        # callback_list = [best_weight]

        # Train neural net
        self.history = self.model.fit(self.train_traj, self.train_ediff, verbose=self.verb, validation_data=(self.val_traj, self.val_ediff), batch_size=config.batch_size, epochs=config.num_epochs, callbacks=callback_list)

        self.reload_weights(config)

    def reload_weights(self, config):
        self.model.load_weights(config.weightfile)

    def compute_metrics(self):
        self.train_pred = self.model_energy.predict(self.train_cg).ravel()
        self.test_pred = self.model_energy.predict(self.test_cg).ravel()
        self.train_pred_diff = self.model.predict(self.train_traj).ravel()
        self.test_pred_diff = self.model.predict(self.test_traj).ravel()

        self.train_mse = np.mean(np.square(self.train_pred - np.mean(self.train_pred) + self.avg_e - self.train_e))
        self.test_mse = np.mean(np.square(self.test_pred - np.mean(self.test_pred) + self.avg_e - self.test_e))

        self.train_mse_diff = np.mean(np.square(self.train_pred_diff - self.train_ediff))
        self.test_mse_diff = np.mean(np.square(self.test_pred_diff - self.test_ediff))

        if self.exact_cg:
            self.compute_cg_metrics()

    def compute_cg_metrics(self):
        self.train_cg_mse = np.mean(np.square(self.train_pred - np.mean(self.train_pred) + self.avg_cg_e - self.train_cg_e))
        self.test_cg_mse = np.mean(np.square(self.test_pred - np.mean(self.test_pred) + self.avg_cg_e - self.test_cg_e))

        self.train_cg_mse_diff = np.mean(np.square(self.train_pred_diff - self.train_cg_ediff))
        self.test_cg_mse_diff = np.mean(np.square(self.test_pred_diff - self.test_cg_ediff))

        self.noise = self.exp_ediff - self.cg_exp_ediff
        self.noise_var = np.var(self.noise)

    def print_metrics(self):
        print()
        print("Metrics:")
        print()
        print("Average energy of samples:", self.avg_e)
        print("Average exp energy of samples:", self.avg_exp_e)
        print()
        print("Train MSE against shifted instantaneous E:", self.train_mse)
        print("Test MSE against shifted instantaneous E:", self.test_mse)
        print()
        print("Train MSE against instantaneous E diff:", self.train_mse_diff)
        print("Test MSE against instantaneous E diff:", self.test_mse_diff)
        if self.exact_cg:
            self.print_cg_metrics()
        print()

    def print_cg_metrics(self):
        print()
        print("CG Metrics:")
        print()
        print("Average CG energy of samples:", self.avg_cg_e)
        print("Average exp CG energy of samples:", self.avg_exp_cg_e)
        print()
        print("Train MSE against shifted CG E:", self.train_cg_mse)
        print("Test MSE against shifted CG E:", self.test_cg_mse)
        print()
        print("Train MSE against CG E diff:", self.train_cg_mse_diff)
        print("Test MSE against CG E diff:", self.test_cg_mse_diff)
        print()
        print("Noise variance:", self.noise_var)

    def graph_loss(self, config):
        plt.figure(1)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.plot(self.history.history['loss'] - np.min(self.history.history['loss']), '.-', label = "loss")
        plt.plot(self.history.history['val_loss'] - np.min(self.history.history['val_loss']), '.-', label = "val_loss")
        plt.legend()
        plt.savefig(config.lossfile, bbox_inches="tight")
