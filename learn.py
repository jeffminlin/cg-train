import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils import plot_model

import convising.train as tr
import convising.mcmc as mc


def run(L):

    config_deep = tr.Config()
    config_deep.L = L
    config_deep.dense_nodes = [20, 20, 3]
    config_deep.num_gpus = 1
    config_deep.verb = 0
    config_deep.model = 'deep_conv'
    config_deep.activ_fcn = 'elu'
    config_deep.conv_activ = 'log_cosh'
    config_deep.refresh_config()

    deep_conv = tr.ConvIsing(config_deep)

    deep_conv.create_cg_dataset(config_deep)
    deep_conv.load_dataset(config_deep)
    # deep_conv.run_model(config_deep)
    deep_conv.reload_weights(config_deep)

    deep_conv.compute_metrics(config_deep)
    deep_conv.print_metrics()
    # deep_conv.graph_loss(config_deep)

    # config_linear = tr.Config()
    # config_linear.L = L
    # config_linear.model = 'linear_basis'
    # config_linear.refresh_config()

    # linear_mod = tr.ConvIsing(config_linear)
    # linear_mod.load_dataset(config_linear)
    # linear_mod.compute_metrics(config_linear)
    # linear_mod.print_metrics()


def compute_iac(deep_conv, config):

    ss_traj = mc.gen_samples(deep_conv.model.predict, int(1e4), 1, np.random.choice([-1, 1], size=(100, config.cgL*config.cgL)), 5000)
    e_traj = deep_conv.model_energy.predict(ss_traj.reshape([-1, config.cgL, config.cgL])).ravel()

    kappa = 250
    plotname = "".join(["./figs/autocorr", config.filepath, ".png"])
    IAC, _ = mc.plot_iac([e_traj], ["Standard M-H"], "".join(["Metropolis-Hastings, L = ", str(config.L), ", CG L = ", str(config.cgL)]), kappa, plotname)
    print("Integrated autocorrelation, kappa =", kappa, ":", IAC)


def compare_observables(deep_conv, config, num_samples, num_chains, batch_size, skip):

    obs_model = mc.Observables(deep_conv.model.predict, config.cgL, num_samples, num_chains, 5000, batch_size, skip)
    obs_model.metrop_par(batch_size)
    print()
    print("Samples generated using learned model")
    obs_model.print_observables()
    print()

    obs_samples = mc.Observables(deep_conv.model.predict, config.cgL, num_samples, num_chains, 5000, batch_size, skip)
    with h5py.File(config.datafile, "r") as dset:
        obs_samples.avgs, obs_samples.vars = obs_samples.compute_observables(dset["".join(["cgimage_", config.cg_method, str(config.cg_factor)])], batch_size)
    obs_samples.num_recorded = 5e6
    print()
    print("Samples generated with Swendsen-Wang")
    obs_samples.print_observables()
    print()


def main():

    run(16)


if __name__ == '__main__':

    main()
