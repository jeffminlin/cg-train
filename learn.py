import datetime
import os
import time

import h5py
import matplotlib

matplotlib.use("Agg")  # matplotlib backend for HPC
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import convising.data as data
import convising.layers as isinglayers
import convising.mcmc as mcmc
import convising.models as models
import convising.train as train


def run(config_ising, config_train, datafile, logdir, cg_ref_file=None):

    print("Ising configuration:", config_ising)
    print("Train configuration:", config_train)

    # Make datasets
    data.create_cg_dataset(
        config_ising, datafile, 0, 3, overwrite=True, cg_ref_file=cg_ref_file
    )
    if cg_ref_file:
        datasets, labels, exact_labels = data.load_datasets(
            datafile,
            config_ising,
            config_train,
            shuffle=True,
            cg_level_start=1,
            cg_level_end=2,
            exact=True,
        )
    else:
        exact_labels = None
        datasets, labels = data.load_datasets(
            datafile,
            config_ising,
            config_train,
            shuffle=True,
            cg_level_start=1,
            cg_level_end=2,
            exact=False,
        )
    # print some information about the data
    print("Number of training samples:", len(datasets[1]["train"][0]))
    print("Number of validation samples:", len(datasets[1]["val"][0]))
    print("Number of testing samples:", len(datasets[1]["test"][0]))

    loop_out = training_loop(
        config_ising, config_train, datasets, labels, logdir, exact_labels
    )

    return loop_out


def training_loop(
    config_ising, config_train, datasets, labels, logdir, exact_labels=None
):

    # Train
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    with strategy.scope():
        deep_model = models.ModelGroup(config_ising, config_train)
        optimizer_adam = tf.keras.optimizers.Adam()
        deep_model.ediff.compile(loss="mse", optimizer=optimizer_adam)

    adam_logdir = os.path.join(logdir, "adam")
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
    train_out = train.train_and_save(
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

    return train_out, deep_model


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
        train_out, _ = training_loop(
            config_ising,
            config_train,
            datasets,
            labels,
            os.path.join(logdir, str(keep_data)),
            exact_labels=exact_labels,
        )
        cg_losses.append(train_out[2]["test"]["loss"])

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
        lambda data: model_group.ediff.predict_on_batch(data),
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
    model_group,
    config_ising,
    datafile,
    num_samples,
    num_chains,
    batch_size,
    skip,
    logdir,
):

    cgL = int(config_ising["L"] / config_ising["cg_factor"])

    obs_model = mcmc.Observables(
        lambda data: model_group.ediff.predict_on_batch(data).numpy(),
        cgL,
        num_samples,
        num_chains,
        50 * skip,
        batch_size,
        skip,
    )
    obs_model.metrop_par(batch_size)
    print()
    print("Samples generated using learned model")
    observed = obs_model.save_observables(logdir, "deep_model_obs.json")
    print(observed)

    obs_samples = mcmc.Observables(
        lambda data: None, cgL, num_samples, num_chains, 5 * skip, batch_size, skip
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
    observed = obs_samples.save_observables(logdir, "sw_obs.json")
    print(observed)


def test_majority_rule_without_blocking():

    config_ising = {"L": 4, "beta": 0.4406868, "cg_method": "maj", "cg_factor": 1}
    config_train = {
        "keep_data": 1.0,
        "train_split": 9.0 / 10.0,
        "val_split": 1.0 / 9.0,
        "batch_size": 25000,  # Should use one GPU's worth of memory efficiently
        "nepochs": 1000,
        "patience": 20,
        "verbosity": 1,
        "conv_activation": "log_cosh",
        "dense_activation": "elu",
        "kernel_size": 3,
        "nfilters": 32,
        "dense_nodes": [20, 20, 3],
    }
    today = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("logs", today)
    datapath = os.path.join(os.curdir, "data")

    start = time.perf_counter()
    start_cpu = time.process_time()
    datafile = os.path.join(
        datapath, "L{0:d}b{1:.4e}.h5".format(config_ising["L"], config_ising["beta"])
    )

    data.create_cg_dataset(config_ising, datafile, 0, 1, overwrite=True)
    datasets, labels = data.load_datasets(
        datafile, config_ising, config_train, cg_level_start=0, cg_level_end=1
    )
    training_loop(
        config_ising, config_train, datasets, labels, logdir, exact_labels=labels
    )
    interm = time.perf_counter()
    interm_cpu = time.process_time()
    print("Training the model")
    print("Elapsed time:", interm - start)
    print("CPU time:", interm_cpu - start_cpu)


def main():

    config_ising = {"L": 4, "beta": 0.4406868, "cg_method": "maj", "cg_factor": 2}
    config_train = {
        "keep_data": 1.0,
        "train_split": 9.0 / 10.0,
        "val_split": 1.0 / 9.0,
        "batch_size": 25000,  # Should use one GPU's worth of memory efficiently
        "nepochs": 1000,
        "patience": 20,
        "verbosity": 1,
        "conv_activation": "log_cosh",
        "dense_activation": "elu",
        "kernel_size": 3,
        "nfilters": 32,
        "dense_nodes": [20, 20, 3],
    }
    today = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("logs", today)
    datapath = os.path.join(os.curdir, "data")

    start = time.perf_counter()
    start_cpu = time.process_time()
    datafile = os.path.join(
        datapath, "L{0:d}b{1:.4e}.h5".format(config_ising["L"], config_ising["beta"])
    )

    for L, skip in [(8, 100), (16, 1000)]:
        start = time.perf_counter()
        start_cpu = time.process_time()
        config_ising["L"] = L
        datafile = os.path.join(
            datapath,
            "L{0:d}b{1:.4e}.h5".format(config_ising["L"], config_ising["beta"]),
        )
        logdir_L = os.path.join(logdir, "L" + str(L))
        model = run(
            config_ising, config_train, datafile, logdir_L
        )[1]
        interm = time.perf_counter()
        interm_cpu = time.process_time()
        print("Loading time for model:")
        print("Elapsed time:", interm - start)
        print("CPU time:", interm_cpu - start_cpu)
        compare_observables(
            model, config_ising, datafile, int(1e6), 25000, 1000, skip, logdir_L
        )
        print("MCMC time:")
        print("Elapsed time:", time.perf_counter() - interm)
        print("CPU time:", time.process_time() - interm_cpu)


if __name__ == "__main__":

    main()
