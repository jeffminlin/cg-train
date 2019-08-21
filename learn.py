import os
import datetime
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
    print("Train configuration:", config_train)

    # Make datasets
    datafile = os.path.join(
        datapath, "L{0:d}b{1:.4e}.h5".format(L, config_ising["beta"])
    )
    cg_ref_file = os.path.join(
        datapath,
        "L{0:d}b{1:.4e}_cgdeci2.h5".format(
            int(L / config_ising["cg_factor"]), config_ising["beta"]
        ),
    )
    data.create_cg_dataset(config_ising, datafile, 0, 2)
    datasets, labels, exact_labels = data.load_datasets(
        datafile, config_ising, config_train, cg_ref_file=cg_ref_file, shuffle=False
    )

    # Train
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        deep_model = models.ModelGroup(config_ising, config_train)
        deep_logdir = os.path.join(logdir, "deep")

        train.train_and_save(
            deep_model,
            datasets,
            labels,
            config_train,
            deep_logdir,
            exact_labels=exact_labels,
        )

        modelpath = os.path.join(logdir, "model")
        tf.keras.experimental.export_saved_model(deep_model.energy, modelpath)


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

    L = 4
    today = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("logs", today)
    datapath = os.path.join(os.curdir, "data")
    run(L, datapath, logdir)


if __name__ == "__main__":

    main()
