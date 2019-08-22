import os
import datetime
import sys
import numpy as np
import h5py
import tensorflow as tf

import convising.data as data
import convising.train as train
import convising.models as models
import convising.mcmc as mcmc


def run(L, datapath, logdir):

    # Setup configurations
    config_ising = train.create_default_config_ising()
    config_ising["L"] = L
    config_ising["cg_method"] = "deci"
    print("Ising configuration:", config_ising)
    config_train = train.create_default_config_train()
    config_train["verbosity"] = 1
    config_train["patience"] = 50
    if len(sys.argv) > 1:
        config_train["batch_size"] = int(sys.argv[1])
    print("Train configuration:", config_train)

    # Make datasets
    datafile = os.path.join(
        datapath, "L{0:d}b{1:.4e}.h5".format(L, config_ising["beta"])
    )
    # data.create_cg_dataset(config_ising, datafile, 0, 2)
    datasets, labels = data.load_datasets(
        datafile, config_ising, config_train, shuffle=False
    )
    # print some information about the data
    print("Number of training samples:", len(datasets[1]["train"][0]))
    print("Number of validation samples:", len(datasets[1]["val"][0]))
    print("Number of testing samples:", len(datasets[1]["test"][0]))

    # Train
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    config_train["batch_size"] *= strategy.num_replicas_in_sync
    with strategy.scope():
        deep_model = models.ModelGroup(config_ising, config_train)
        deep_model.ediff.compile(loss="mse", optimizer="Adam")

    deep_logdir = os.path.join(logdir, "deep")
    train.train_and_save(deep_model, datasets, labels, config_train, deep_logdir)


# NEED TO UPDATE
def compute_iac(model_group, config):

    ss_traj = mcmc.gen_samples(
        model_group.ediff.predict,
        int(1e4),
        1,
        np.random.choice([-1, 1], size=(100, config.cgL * config.cgL)),
        5000,
    )
    e_traj = model_group.energy.predict(
        ss_traj.reshape([-1, config.cgL, config.cgL])
    ).ravel()

    kappa = 250
    plotname = "".join(["./figs/autocorr", config.filepath, ".png"])
    IAC, _ = mcmc.plot_iac(
        [e_traj],
        ["Standard M-H"],
        "".join(
            ["Metropolis-Hastings, L = ", str(config.L), ", CG L = ", str(config.cgL)]
        ),
        kappa,
        plotname,
    )
    print("Integrated autocorrelation, kappa =", kappa, ":", IAC)


# NEED TO UPDATE
def compare_observables(model_group, config, num_samples, num_chains, batch_size, skip):

    obs_model = mcmc.Observables(
        model_group.ediff.predict,
        config.cgL,
        num_samples,
        num_chains,
        5000,
        batch_size,
        skip,
    )
    obs_model.metrop_par(batch_size)
    print()
    print("Samples generated using learned model")
    obs_model.print_observables()
    print()

    obs_samples = mcmc.Observables(
        model_group.ediff.predict,
        config.cgL,
        num_samples,
        num_chains,
        5000,
        batch_size,
        skip,
    )
    with h5py.File(config.datafile, "r") as dset:
        obs_samples.avgs, obs_samples.vars = obs_samples.compute_observables(
            dset["".join(["cgimage_", config.cg_method, str(config.cg_factor)])],
            batch_size,
        )
    obs_samples.num_recorded = 5e6
    print()
    print("Samples generated with Swendsen-Wang")
    obs_samples.print_observables()
    print()


def main():

    L = 16
    today = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("logs", today)
    datapath = os.path.join(os.curdir, "data")
    run(L, datapath, logdir)


if __name__ == "__main__":

    main()
