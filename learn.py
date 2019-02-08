import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils import plot_model

import convising.train as tr
import convising.mcmc as mc


def run():

    config = tr.Config()
    config.L = 4
    config.conv_activ = 'log_cosh'
    config.exact_cg = False
    config.keep = 1.0
    config.batch_size = 20000
    config.num_epochs = 5
    config.verb = 1
    config.refresh_config()
    deep_conv = tr.ConvIsing(config)

    try:
        deep_conv.create_cg_dataset(config)
    except RuntimeError as error:
        print(error)
        print("Coarse-grained data likely already created")
    else:
        print("Coarse-grained data created")
    deep_conv.load_dataset(config)
    deep_conv.run_model(config)
    deep_conv.compute_metrics()
    deep_conv.print_metrics()
    deep_conv.graph_loss(config)


def compute_iac(deep_conv, config):

    ss_traj = mc.gen_samples(deep_conv.model.predict, int(1e4), 1, np.random.choice([-1, 1], size=(100, config.cgL*config.cgL)), 5000)
    e_traj = deep_conv.model_energy.predict(ss_traj.reshape([-1, config.cgL, config.cgL])).ravel()

    kappa = 250
    plotname = "".join(["./figs/autocorr", config.filepath, ".png"])
    IAC, autocorr_fun = mc.plot_iac([e_traj], ["Standard M-H"], "".join(["Metropolis-Hastings, L = ", str(config.L), ", CG L = ", str(config.cgL)]), kappa, plotname)
    print("Integrated autocorrelation, kappa =", kappa, ":", IAC)


def compare_observables(deep_conv, config, num_samples, num_chains, skip):

    obs_model = mc.Observables(deep_conv.model.predict, config.cgL, num_samples, num_chains, 5000, skip)
    obs_model.metrop_par(1000)
    print()
    print("Samples generated using learned model")
    obs_model.print_observables()
    print()

    obs_samples = mc.Observables(deep_conv.model.predict, config.cgL, num_samples, num_chains, 5000, skip)
    with h5py.File(config.datafile, "r") as dset:
        obs_samples.avgs, obs_samples.vars = obs_samples.compute_observables(dset["".join(["cgimage_", config.cg_method, str(config.cg_factor)])], 1000)
    obs_samples.num_recorded = 5e6
    print()
    print("Samples generated with Swendsen-Wang")
    obs_samples.print_observables()
    print()


def main():

    config = tr.Config()
    config.L = 16
    config.num_gpus = 4
    config.conv_activ = 'log_cosh'
    config.refresh_config()
    deep_conv = tr.ConvIsing(config)

    deep_conv.create_cg_dataset(config)
    deep_conv.load_dataset(config)
    deep_conv.run_model(config)
    deep_conv.reload_weights(config)

    deep_conv.compute_metrics(config)
    deep_conv.print_metrics()
    deep_conv.graph_loss(config)


if __name__ == '__main__':

    main()
