"""Module for training"""
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.engine.topology import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
import tensorflow as tf

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
        num_gpus (type): Description of parameter `num_gpus`.
        verb (int): Verbosity parameter for keras fit
                (0 = no output, 1 = detailed output, 2 = some output)

        exact_cg (bool): Whether or not to use any data on exact cg values

        model (str): Can be either 'deep_conv' for the convolutional model or 'linear_basis' for the fixed basis model
        activ_fcn (str): Sets the activation functions on the dense layers, e.g. relu or elu
        w_size (int): Filter window size, must be odd if nfsym = 'all' or 'z2'
        alpha (int): Number of filters
        conv_activ (str): Choice of activation function on the
            convolutional layer, either 'linear' or 'log_cosh'
        dense_nodes (int): List of integers describing the number of nodes in
                each dense layer
        nfsym (str): Symmetries to enforce, can be 'none', 'z2', 'd4', or 'all'

    Methods:
        refresh_config: Must be run if the configuration changes after it is initialized, in order to fix filenames

    """
    def __init__(self, L = 8, beta = .4406868, cg_method = "maj", cg_factor = 2):
        self.L = L
        self.beta = beta
        self.cg_method = cg_method
        self.cg_factor = cg_factor

        self.keep = 1.0
        self.train_split = 9./10.
        self.val_split = 1./9.
        self.batch_size = 20000
        self.num_epochs = 1000
        self.num_gpus = 1
        self.verb = 0

        # Can set this to be True if L = 4 and the cg reference data is generated
        self.exact_cg = False

        self.model = 'deep_conv'
        self.activ_fcn = 'elu'
        self.w_size = 3
        self.alpha = 4
        self.conv_activ = 'log_cosh'
        self.dense_nodes = [20, 20, 4]
        self.nfsym = "none"

        self.refresh_config()

    def refresh_config(self):
        self.cgL = int(self.L/self.cg_factor)

        self.filepath_core = "".join(["L{0:d}b{1:.4e}", "a", str(self.alpha), "w", str(self.w_size), self.model, "_", self.conv_activ, "_", self.activ_fcn, "_", *("".join(["n",str(dn)]) for dn in self.dense_nodes), self.cg_method, str(self.cg_factor), "nf", self.nfsym]).format(self.L, self.beta)
        self.weightfile = "".join(["./weights/", self.filepath_core, ".h5"])
        self.weightfile_freeze = "".join(["./weights/f", self.filepath_core, ".h5"])
        self.lossfile = "".join(["./figs/loss", self.filepath_core, ".png"])

        self.datafile = "./data/L{0:d}b{1:.4e}.h5".format(self.L, self.beta)

        self.cgref_datafile = "".join(["./data/L{0:d}b{1:.4e}_cg", self.cg_method, str(self.cg_factor), ".h5"]).format(self.cgL, self.beta)


class MultiGPUCheckpoint(Callback):
    """Save the model, with multiple GPUs (see ModelCheckpoint).

    Args:
        filepath (str): File of saved weights.
        base_model (Model): Keras Model whose weights are being saved.
        monitor (str): Metric to monitor.
        verbose: Whether or not to print information each epoch.
            Can be 0, 1 or 2. Defaults to 0.
        save_best_only (bool): Whether to save the best weights or the most
            recent weights. Defaults to False.
        save_weights_only (bool): Whether to save just the weights or the
            network graph as well. Defaults to False.
        mode (str): If 'save_best_only' is True, decides to save new weights
            based on when the monitored metric is maximized or minimized.
            Defaults to 'auto'.
        period (int): Number of epochs between checkpoints.

    """

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiGPUCheckpoint, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)


def coarse_grain(L, beta, cg_method, cgf, image):

    image = image.reshape([-1,L,L])
    numdat = len(image)
    cgL = int(L/cgf)
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


class GeneratorIsing(Sequence):
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
        self.exp_ediff = dset["".join(["exp_ediff1_", config.cg_method, str(config.cg_factor)])]
        numdat = int(len(self.exp_ediff)*config.keep)
        num_train = int(numdat*config.train_split*(1 - config.val_split))
        num_val = int(numdat*config.train_split*config.val_split)
        num_test = numdat - num_train - num_val
        if set == "train":
            self.idxset = np.arange(num_train)
        elif set == "val":
            self.idxset = np.arange(num_val) + num_train
        else:
            self.idxset = np.arange(num_test) + num_train + num_val
        self.num_samples = len(self.idxset)

        self.image = dset["".join(["cgimage", str(num_cg), "_", config.cg_method, str(config.cg_factor)])][self.idxset,:,:]
        self.imageflip = dset["".join(["cgimageflip", str(num_cg), "_", config.cg_method, str(config.cg_factor)])][self.idxset,:,:]
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
        batch_image = self.image[self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size].tolist(),:,:]
        batch_imageflip = self.imageflip[self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size].tolist(),:,:]
        batch_ediff = self.exp_ediff[self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size].tolist()]

        if self.flip:
            return ([batch_image, batch_imageflip], batch_ediff)
        else:
            return([batch_image], batch_ediff)

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


class ConvIsing:
    """Class for training a neural network to learn the coarse-grained
        Hamiltonian for the Ising model.

    Args:
        config (Config)
            Configuration class

    """

    def __init__(self, config):
        self.L = config.L
        self.beta = config.beta
        self.cg_method = config.cg_method
        self.cgf = config.cg_factor
        self.verb = config.verb
        self.exact_cg = config.exact_cg
        self.create_model(config)

    def create_cg_dataset(self, config, start_cg, end_cg):
        with h5py.File(config.datafile, "r+") as dset:
            if start_cg == 0:
                imagearray = dset["images"][:,:]
            else:
                imagearray = dset["".join(["cgimage", str(start_cg), "_", config.cg_method, str(config.cg_factor)])]

            for flipidx in range(end_cg - start_cg - 1):
                ([cgimage, cgimageflip], exp_ediff) = coarse_grain(self.L, self.beta, self.cg_method, self.cgf, imagearray)

                h5py_create_data_catch(dset, "".join(["cgimage", str(start_cg + flipidx + 1), "_", config.cg_method, str(config.cg_factor)]), data=cgimage, dtype='i1')
                h5py_create_data_catch(dset, "".join(["cgimageflip", str(start_cg + flipidx + 1), "_", config.cg_method, str(config.cg_factor)]), data=cgimageflip, dtype='i1')
                if start_cg == 0 and flipidx == 0:
                    h5py_create_data_catch(dset, "".join(["exp_ediff1_", config.cg_method, str(config.cg_factor)]), data=exp_ediff)

                imagearray = cgimage

    def load_dataset(self, config):
        self.dset = h5py.File(config.datafile, "r")
        if self.exact_cg:
            self.compute_exact_cg(config)

    def compute_exact_cg(self, config):
        self.cgimage = self.dset["".join(["cgimage_", config.cg_method, str(config.cg_factor)])]
        self.cgimageflip = self.dset["".join(["cgimageflip_", config.cg_method, str(config.cg_factor)])]
        self.cgref_dset = h5py.File(config.cgref_datafile, "r")
        self.cgref_ss = self.cgref_dset["images"]
        self.cgref_e = self.cgref_dset["energies"][:]/self.beta

        self.cg_e = lookup_cg_e(self.L, self.beta, self.cgf, self.cgimage, self.cgref_ss, self.cgref_e)
        cg_e_flip = lookup_cg_e(self.L, self.beta, self.cgf, self.cgimageflip, self.cgref_ss, self.cgref_e)
        self.cg_exp_ediff = np.exp(-self.beta*(cg_e_flip - self.cg_e))

    def create_model(self, config):
        K.clear_session()
        conv_activ = config.conv_activ
        activ_fcn = config.activ_fcn
        kninit = 'glorot_normal'

        # pick model
        if config.model == 'linear_basis':
            model_energy = mod.linear_basis(config, kninit)
        else:
            model_energy = mod.deep_conv_e(config, conv_activ, activ_fcn, kninit)

        if config.num_gpus > 1:
            with tf.device('/cpu:0'):
                self.model_energy = model_energy
                self.model = mod.model_e_diff(config, self.model_energy)
        else:
            self.model_energy = model_energy
            self.model = mod.model_e_diff(config, self.model_energy)

        self.model.compile(loss='mean_squared_error', optimizer='Nadam')

    def run_model(self, config, freeze=False):
        train_generator = GeneratorIsing(self.dset, config, 1, shuffle=True, set="train")
        val_generator = GeneratorIsing(self.dset, config, 1, shuffle=True, set="val")

        weightfile = config.weightfile
        if freeze:
            weightfile = config.weightfile_freeze
            for layer in self.model_energy.layers[:-1]:
                layer.trainable = False
        else:
            for layer in self.model_energy.layers:
                layer.trainable = True

        best_weight = MultiGPUCheckpoint(weightfile, self.model_energy,
            monitor="val_loss", save_best_only=True, save_weights_only=True)
        early_stop = EarlyStopping(monitor="val_loss", patience=100,
            mode='min') # Setting patience=config.num_epochs means the
                        # training does not stop early but it restores the
                        # best results (requires Keras >2.2.3, unsure if this
                        # is working)
        callback_list = [best_weight, early_stop]
        # callback_list = [best_weight]

        # Train neural net
        # Using data stream
        if config.num_gpus > 1:
            par_model = multi_gpu_model(self.model, gpus=config.num_gpus)
            par_model.compile(loss='mean_squared_error', optimizer='Nadam')
            self.history = par_model.fit_generator(generator=train_generator,
                                         verbose=self.verb,
                                         validation_data=val_generator,
                                         epochs=config.num_epochs,
                                         callbacks=callback_list,
                                         use_multiprocessing=False,
                                         max_queue_size=20)
        else:
            self.history = self.model.fit_generator(generator=train_generator,
                                         verbose=self.verb,
                                         validation_data=val_generator,
                                         epochs=config.num_epochs,
                                         callbacks=callback_list,
                                         use_multiprocessing=False,
                                         max_queue_size=20)

        self.reload_weights(config)

    def reload_weights(self, config, freeze=False):
        weightfile = config.weightfile
        if freeze:
            weightfile = config.weightfile_freeze
        self.model_energy.load_weights(weightfile)

    def compute_metrics(self, config, start_cg, end_cg):
        train_generator = GeneratorIsing(self.dset, config, 1, shuffle=False, set="train")
        test_generator1 = GeneratorIsing(self.dset, config, 1, shuffle=False, set="test")
        test_generator1_noflip = GeneratorIsing(self.dset, config, start_cg, shuffle=False, set="test", flip=False)
        test_generator2_noflip = GeneratorIsing(self.dset, config, end_cg, shuffle=False, set="test", flip=False)
        self.train_mse_diff = self.model.evaluate_generator(generator=train_generator, use_multiprocessing=False).ravel()
        self.test_mse_diff = self.model.evaluate_generator(generator=test_generator1, use_multiprocessing=False).ravel()

        if self.exact_cg:
            self.compute_cg_metrics()

        nn_basis = Model(inputs = self.model_energy.layers[0].input, outputs = self.model_energy.layers[-2].output)
        nn_fine = nn_basis.predict_generator(generator=test_generator1_noflip, use_multiprocessing=False)
        nn_coarse = nn_basis.predict_generator(generator=test_generator2_noflip, use_multiprocessing=False)

        print("Singular values (fine):", K.eval(tf.svd(nn_fine, compute_uv = False)))
        print("Condition number of nn basis (fine):", np.linalg.cond(nn_fine))
        print("Singular values (coarse):", K.eval(tf.svd(nn_coarse, compute_uv = False)))
        print("Condition number of nn basis (coarse):", np.linalg.cond(nn_coarse))

        coarse_avg = np.average(nn_coarse, axis=0)
        coarse_avg = coarse_avg.reshape(1, -1)
        fine_avg = np.average(nn_fine, axis=0)
        fine_avg = fine_avg.reshape(1, -1)
        Mcc = np.matmul(nn_coarse.transpose(), nn_coarse)/nn_coarse.shape[0] - np.matmul(coarse_avg.transpose(), coarse_avg)
        self.cc_cond = np.linalg.cond(Mcc)
        Mcf = np.matmul(nn_coarse.transpose(), nn_fine)/nn_coarse.shape[0] - np.matmul(coarse_avg.transpose(), fine_avg)

        self.J = np.linalg.lstsq(Mcc, Mcf, rcond=None)[0]
        self.J_eigs, _ = np.linalg.eig(self.J)
        self.criticalexp = np.log(2)/np.log(np.max(np.real(self.J_eigs)))


    def compute_cg_metrics(self, config):
        train_generator = GeneratorIsing(self.dset, config, 1, shuffle=False, set="train")
        test_generator1 = GeneratorIsing(self.dset, config, 1, shuffle=False, set="test")
        self.train_pred_diff = self.model.predict_generator(generator=train_generator, use_multiprocessing=False).ravel()
        self.test_pred_diff = self.model.predict_generator(generator=test_generator1, use_multiprocessing=False).ravel()
        self.train_cg_mse_diff = np.mean(np.square(self.train_pred_diff - self.train_cg_ediff))
        self.test_cg_mse_diff = np.mean(np.square(self.test_pred_diff - self.test_cg_ediff))

        self.noise = self.exp_ediff - self.cg_exp_ediff
        self.noise_var = np.var(self.noise)

    def print_metrics(self):
        print()
        print("Metrics:")
        print()
        print("Train MSE against instantaneous E diff:", self.train_mse_diff)
        print("Test MSE against instantaneous E diff:", self.test_mse_diff)
        if self.exact_cg:
            self.print_cg_metrics()
        print()

        print("Mcc^(-1)Mcf:", self.J)
        print("Eigenvalues:", self.J_eigs)
        print("Condition number of Mcc:", self.cc_cond)
        print("Critical exp:", self.criticalexp)

    def print_cg_metrics(self):
        print()
        print("CG Metrics:")
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
