import numpy as np

from keras.layers import Dense, Lambda, Input, Conv2D, Activation
from keras.layers import Add, Multiply, Subtract, Dot, Reshape
from keras.models import Model
from keras import optimizers, regularizers
from keras.engine.topology import Layer
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import Sequence

from layers import *


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


def lookup_cg_e(L, beta, cgf, cgimage, cgref_ss, cgref_e):

    im_equal = [np.argwhere(np.sum(np.square(cgref_ss - image.ravel()), axis=1) == 0) for image in cgimage]
    cg_e = np.array([cgref_e[idx] for idx in im_equal]).ravel()

    return cg_e


class ConvIsing:

    def __init__(self, config):
        self.L = config.L
        self.beta = config.beta
        self.cg_method = config.cg_method
        self.cgf = config.cg_factor
        self.create_model(config)
        self.create_dataset(config)

    def create_dataset(self, config):
        self.imagearray = np.load(config.imagefile, mmap_mode="r")
        self.earray = np.load(config.efile, mmap_mode="r")/self.beta
        self.cgref_ss = np.load(config.cgref_imagefile, mmap_mode="r")
        self.cgref_e = np.load(config.cgref_efile, mmap_mode="r")/self.beta

        ([image, imageflip], self.exp_ediff) = coarse_grain(self.L, self.beta, self.cg_method, self.cgf, self.imagearray)

        self.cg_e = lookup_cg_e(self.L, self.beta, self.cgf, image, self.cgref_ss, self.cgref_e)
        cg_e_flip = lookup_cg_e(self.L, self.beta, self.cgf, imageflip, self.cgref_ss, self.cgref_e)

        self.cg_exp_ediff = np.exp(-self.beta*(cg_e_flip - self.cg_e))

        self.avg_e = np.mean(self.earray)
        self.avg_cg_e = np.mean(self.cg_e)
        self.avg_exp_e = np.mean(self.exp_ediff)
        self.avg_exp_cg_e = np.mean(self.cg_exp_ediff)

        self.imagearray = self.imagearray.reshape([-1,self.L,self.L])

        (self.train, self.val, self.test) = self.split_dataset(self.imagearray, config)
        (train_image, val_image, test_image) = self.split_dataset(image, config)
        (train_imageflip, val_imageflip, test_imageflip) = self.split_dataset(imageflip, config)
        self.train_traj = [train_image, train_imageflip]
        self.val_traj = [val_image, val_imageflip]
        self.test_traj = [test_image, test_imageflip]

        (self.train_ediff, self.val_ediff, self.test_ediff) = self.split_dataset(self.exp_ediff, config)
        (self.train_e, self.val_e, self.test_e) = self.split_dataset(self.earray, config)
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

        activation_fcn = 'tanh'
        M_in = Input(shape=(None,None))
        M_pad = PeriodicPad2D(name='pad', pad_size=config.w_size-1)(M_in)
        M_conv = Conv2DNFSym(config.alpha, [config.w_size,config.w_size], strides=(1,1), activation='linear', padding='valid', use_bias=False, nfsym=config.nfsym)(M_pad)
        if config.nfsym == 'z2' or config.nfsym == 'all':
            M_conv = Lambda(lambda x: K.expand_dims(x[0])*x[1])([M_in,M_conv])
        if len(config.dense_nodes) > 1:
            M_fc = Dense(config.dense_nodes[0], activation=activation_fcn, use_bias=True)(M_conv)
            for nodenum in config.dense_nodes[1:]:
                M_fc = Dense(nodenum, activation=activation_fcn, use_bias=False)(M_fc)
        else:
            M_fc = Dense(config.dense_nodes[0], activation=activation_fcn, use_bias=True)(M_conv)
        M_sum = Lambda(lambda x: K.sum(x,axis=[1,2]), name='filter_e')(M_fc)
        M_lincomb = Dense(1, activation='linear', use_bias=False)(M_sum)
        self.model_energy = Model(inputs=M_in, outputs=M_lincomb)

        M_in_concat = [Input(shape=(None,None)),Input(shape=(None,None))]
        M_energy_diff = Subtract(name='energy_diff')([self.model_energy(M_in_concat[1]), self.model_energy(M_in_concat[0])])
        M_exp = Lambda(lambda x: K.exp(-config.beta*x), name='exp')(M_energy_diff)
        self.model = Model(inputs=M_in_concat, outputs=M_exp)

    def run_model(self, config):
        self.model.compile(loss="mean_squared_error", optimizer="Nadam")

        best_weight = ModelCheckpoint(config.weightfile, monitor="val_loss", save_best_only=True, save_weights_only=True)
        # early_stop = EarlyStopping(monitor="val_loss", patience=config.num_epochs, restore_best_weights=True) # Setting patience=config.num_epochs means the training does not stop early but it restores the best results (requires Keras 2.2.3, unsure if this is working)
        # callback_list = [best_weight,early_stop]
        callback_list = [best_weight]

        # Train neural net
        if config.noise:
            # Noisy labels
            self.model.fit(self.train_traj, self.train_ediff, verbose=1, validation_data=(self.val_traj, self.val_ediff), batch_size=config.batch_size, epochs=config.num_epochs, callbacks=callback_list)
        else:
            # Noiseless labels
            self.model.fit(self.train_traj, self.train_cg_ediff, verbose=1, validation_data=(self.val_traj, self.val_cg_ediff), batch_size=config.batch_size, epochs=config.num_epochs, callbacks=callback_list)

        self.model.load_weights(config.weightfile)

    def compute_metrics(self):
        self.train_pred = self.model_energy.predict(self.train).flatten()
        self.test_pred = self.model_energy.predict(self.test).flatten()
        self.train_mse = np.mean(np.square(self.train_pred - np.mean(self.train_pred) + self.avg_e - self.train_e))
        self.test_mse = np.mean(np.square(self.test_pred - np.mean(self.test_pred) + self.avg_e - self.test_e))

        self.train_cg_mse = np.mean(np.square(self.train_pred - np.mean(self.train_pred) + self.avg_cg_e - self.train_cg_e))
        self.test_cg_mse = np.mean(np.square(self.test_pred - np.mean(self.test_pred) + self.avg_cg_e - self.test_cg_e))

        self.train_mse_diff = self.model.evaluate(self.train_traj, self.train_ediff, verbose=1)
        self.test_mse_diff = self.model.evaluate(self.test_traj, self.test_ediff, verbose=1)

        self.train_cg_mse_diff = self.model.evaluate(self.train_traj, self.train_cg_ediff, verbose=1)
        self.test_cg_mse_diff = self.model.evaluate(self.test_traj, self.test_cg_ediff, verbose=1)

        self.noise_var = np.var(self.exp_ediff - self.cg_exp_ediff)

    def print_metrics(self):
        print()
        print("Metrics:")
        print()
        print("Average energy of samples:", self.avg_e)
        print("Average CG energy of samples:", self.avg_cg_e)
        print("Average exp energy of samples:", self.avg_exp_e)
        print("Average exp CG energy of samples:", self.avg_exp_cg_e)
        print()
        print("Train MSE against shifted instantaneous E:", self.train_mse)
        print("Test MSE against shifted instantaneous E:", self.test_mse)
        print()
        print("Train MSE against shifted CG E:", self.train_cg_mse)
        print("Test MSE against shifted CG E:", self.test_cg_mse)
        print()
        print("Train MSE against instantaneous E diff:", self.train_mse_diff)
        print("Test MSE against instantaneous E diff:", self.test_mse_diff)
        print()
        print("Train MSE against CG E diff:", self.train_cg_mse_diff)
        print("Test MSE against CG E diff:", self.test_cg_mse_diff)
        print()
        print("Noise variance:", self.noise_var)
        print()
