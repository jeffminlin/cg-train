import os
import datetime
import numpy as np
import h5py
import matplotlib

matplotlib.use("Agg")  # matplotlib backend for HPC
import matplotlib.pyplot as plt
import tensorflow as tf

import convising.data as data
import convising.train as train
import convising.models as models
import convising.mcmc as mcmc


def run(config_ising, config_train, datapath, logdir):

    print("Ising configuration:", config_ising)
    print("Train configuration:", config_train)

    # Make datasets
    datafile = os.path.join(
        datapath, "L{0:d}b{1:.4e}.h5".format(config_ising["L"], config_ising["beta"])
    )
    data.create_cg_dataset(config_ising, datafile, 0, 3)
    datasets, labels = data.load_datasets(
        datafile,
        config_ising,
        config_train,
        shuffle=True,
        cg_level_start=1,
        cg_level_end=2,
    )
    # print some information about the data
    print("Number of training samples:", len(datasets[1]["train"][0]))
    print("Number of validation samples:", len(datasets[1]["val"][0]))
    print("Number of testing samples:", len(datasets[1]["test"][0]))

    training_loop(config_ising, config_train, datasets, labels, logdir)


def training_loop(
    config_ising, config_train, datasets, labels, logdir, exact_labels=None
):

    # Train
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    config_train["batch_size"] *= strategy.num_replicas_in_sync
    with strategy.scope():
        deep_model = models.ModelGroup(config_ising, config_train)
        optimizer_adam = tf.keras.optimizers.Adam()
        deep_model.ediff.compile(loss="mse", optimizer=optimizer_adam)

    adam_logdir = os.path.join(logdir, "adam")
    config_train["patience"] = 20
    train.train_and_save(
        deep_model,
        datasets,
        labels,
        config_ising,
        config_train,
        adam_logdir,
        cg_level_start=1,
        cg_level_end=2,
        exact_labels=exact_labels,
    )

    with strategy.scope():
        optimizer_sgd = tf.keras.optimizers.SGD(learning_rate=0.00005, momentum=0.01)
        deep_model.ediff.compile(loss="mse", optimizer=optimizer_sgd)

    deep_model.ediff.load_weights(os.path.join(adam_logdir, "weights.h5"))
    sgd_logdir = os.path.join(logdir, "sgd")
    config_train["patience"] = 20
    history, metrics, exact_metrics = train.train_and_save(
        deep_model,
        datasets,
        labels,
        config_ising,
        config_train,
        sgd_logdir,
        cg_level_start=1,
        cg_level_end=2,
        exact_labels=exact_labels,
    )

    if exact_labels:
        return history, metrics, exact_metrics

    return history, metrics


def cg_loss_vs_nsamples(config_ising, config_train, keep_list, datafile, logdir):

    cg_losses = []
    for keep_data in keep_list:
        config_train["keep_data"] = keep_data
        print("Keeping", keep_data, "of the data")
        print()
        datasets, labels, exact_labels = data.load_datasets(
            datafile,
            config_ising,
            config_train,
            shuffle=True,
            cg_level_start=1,
            cg_level_end=2,
            exact=True,
        )
        history, metrics, exact_metrics = training_loop(
            config_ising,
            config_train,
            datasets,
            labels,
            os.path.join(logdir, str(keep_data)),
            exact_labels=exact_labels,
        )
        cg_losses.append(exact_metrics["test"]["loss"])

    print("Percentages of data kept:", keep_list)
    print("Exact losses:", cg_losses)

    plt.figure()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("number of training samples")
    plt.ylabel("Test MSE against exact coarse-grained energy differences")
    plt.plot(keep_list * len(datasets[1]["train"]), cg_losses, ".")
    plt.savefig(
        os.path.join(logdir, "ising_sample_size_vs_CG_loss.png"), bbox_inches="tight"
    )
    plt.close()


def compute_iac(model_group, config_ising, plotpath):

    cgL = int(config_ising["L"] / config_ising["cg_factor"])

    ss_traj = mcmc.gen_samples(
        model_group.ediff.predict,
        int(1e4),
        1,
        np.random.choice([-1, 1], size=(100, cgL * cgL)),
        5000,
    )
    e_traj = model_group.energy.predict(ss_traj.reshape([-1, cgL, cgL])).ravel()

    kappa = 250
    IAC, _ = mcmc.plot_iac(
        [e_traj],
        ["Standard M-H"],
        "Metropolis-Hastings, L = " + str(config_ising["L"]) + ", CG L = " + str(cgL),
        kappa,
        plotpath,
    )
    print("Integrated autocorrelation, kappa =", kappa, ":", IAC)


def compare_observables(
    model_group, config_ising, datafile, num_samples, num_chains, batch_size, skip
):

    cgL = int(config_ising["L"] / config_ising["cg_factor"])

    obs_model = mcmc.Observables(
        model_group.ediff.predict, cgL, num_samples, num_chains, 5000, batch_size, skip
    )
    obs_model.metrop_par(batch_size)
    print()
    print("Samples generated using learned model")
    obs_model.print_observables()
    print()

    obs_samples = mcmc.Observables(
        model_group.ediff.predict, cgL, num_samples, num_chains, 5000, batch_size, skip
    )
    with h5py.File(datafile, "r") as dset:
        group = "/".join(
            ["1", config_ising["cg_method"], str(config_ising["cg_factor"])]
        )
        obs_samples.avgs, obs_samples.variances = obs_samples.compute_observables(
            dset[group]["images"], batch_size
        )
        obs_samples.num_recorded = len(dset[group]["images"])
    print()
    print("Samples generated with Swendsen-Wang")
    obs_samples.print_observables()
    print()


def main():

    config_ising = {"L": 4, "beta": 0.4406868, "cg_method": "deci", "cg_factor": 2}
    config_train = {
        "keep_data": 1.0,
        "train_split": 9.0 / 10.0,
        "val_split": 1.0 / 9.0,
        "batch_size": 2500,  # Should use one GPU's worth of memory efficiently
        "nepochs": 1000,
        "patience": 100,
        "verbosity": 0,
        "conv_activation": "log_cosh",
        "dense_activation": "elu",
        "kernel_size": 3,
        "nfilters": 32,
        "dense_nodes": [20, 20, 3],
    }
    today = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("logs", today)

    datapath = os.path.join(os.curdir, "data")
    # Make datasets
    datafile = os.path.join(
        datapath, "L{0:d}b{1:.4e}.h5".format(config_ising["L"], config_ising["beta"])
    )
    cg_ref_file = os.path.join(
        datapath,
        "L{0:d}b{1:.4e}_cgdeci2.h5".format(
            int(config_ising["L"] / config_ising["cg_factor"]), config_ising["beta"]
        ),
    )
    data.create_cg_dataset(config_ising, datafile, 0, 2, cg_ref_file=cg_ref_file)

    keep_list = np.power(10.0, np.linspace(-3.0, 0.0, num=40))
    cg_loss_vs_nsamples(config_ising, config_train, keep_list, datafile, logdir)


if __name__ == "__main__":

    main()
