import numpy as np
import pylab as pyl
import tensorflow as tf

import neuralnets as nn


class Config:

    def __init__(self):
        self.L = 4
        self.beta = .4406868
        self.imagefile = "data/ssL{0:d}b{1:.4e}.npy".format(self.L,self.beta)
        self.efile = "data/EL{0:d}b{1:.4e}.npy".format(self.L,self.beta)
        self.cg_method = "deci"
        self.cg_factor = 2
        self.cgref_imagefile = "".join(["data/ssL{0:d}b{1:.4e}_cg", self.cg_method, ".npy"]).format(int(self.L/self.cg_factor),self.beta)
        self.cgref_efile = "".join(["data/EL{0:d}b{1:.4e}_cg", self.cg_method, ".npy"]).format(int(self.L/self.cg_factor),self.beta)

        self.train_split = 3./5.
        self.val_split = 1./3.
        self.dim = (4,4)
        self.num_channels = 1
        self.shuffle = True
        self.batch_size = 4000
        self.num_epochs = 2000
        self.noise = True

        self.w_size = 3
        self.alpha = 4
        self.dense_nodes = [40, 40]
        self.nfsym = "none"

        self.filepath_rename()

    def filepath_rename(self):
        self.weightfile = "".join(["./weights/", "a", str(self.alpha), "w", str(self.w_size), *("".join(["n",str(dn)]) for dn in self.dense_nodes), ".hdf5"])


def check_optimizer(config):

    deep_conv = nn.ConvIsing(config)

    config.noise = True
    deep_conv.run_model(config)
    deep_conv.compute_metrics()
    deep_conv.print_metrics()


def main():

    # tf.logging.set_verbosity(tf.logging.ERROR)
    config = Config()
    check_optimizer(config)


if __name__ == '__main__':

    main()
