"""Module for training"""
import os
import json

import h5py
import matplotlib

matplotlib.use("Agg")  # matplotlib backend for HPC
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def create_default_config_ising():

    return {"L": 8, "beta": 0.4406868, "cg_method": "maj", "cg_factor": 2}


def create_default_config_train():

    config = {
        "keep_data": 1.0,
        "train_split": 9.0 / 10.0,
        "val_split": 1.0 / 9.0,
        "batch_size": 20000,
        "nepochs": 1000,
        "verbosity": 0,
        "conv_activation": "log_cosh",
        "dense_activation": "elu",
        "kernel_size": 3,
        "nfilters": 4,
        "dense_nodes": [20, 20, 4],
    }

    return config


def train(model_group, datasets, labels, config_train, logdir, freeze=False):

    weightfile = os.path.join(logdir, "weights.h5")
    if freeze:
        weightfile = os.path.join(logdir, "weights_frozen.h5")
        for layer in model_group.energy.layers[:-1]:
            layer.trainable = False
    else:
        for layer in model_group.energy.layers:
            layer.trainable = True

    best_weight = tf.keras.callbacks.ModelCheckpoint(
        weightfile, monitor="val_loss", save_best_only=True, save_weights_only=True
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=100, restore_best_weights=True
    )
    callback_list = [best_weight, early_stop]

    # Train neural net
    model_group.ediff.compile(loss="mean_squared_error", optimizer="Nadam")
    history = model_group.ediff.fit(
        datasets["train"],
        labels["train"],
        verbose=config_train["verbosity"],
        validation_split=config_train["val_split"],
        epochs=config_train["nepochs"],
        callbacks=callback_list,
    )

    return history


def train_and_save(
    model_group,
    datasets,
    labels,
    config_train,
    logdir,
    cg_level_start=1,
    cg_level_end=2,
    freeze=False,
    exact_labels=None,
):

    history = train(model_group, datasets, labels, config_train, logdir, freeze)
    save_loss(history, logdir)
    compute_rg_metrics(model_group, datasets, labels, cg_level_start, cg_level_end)
    if exact_labels:
        compute_exact_cg_metrics(model_group, datasets, labels, exact_labels, logdir)


def compute_stat_avg(first_samples, second_samples):

    average_first = np.average(first_samples, axis=0).reshape(1, -1)
    average_second = np.average(second_samples, axis=0).reshape(1, -1)
    M = np.matmul(first_samples.transpose(), second_samples) / second_samples.shape[0]
    M = M - np.matmul(average_first.transpose(), average_second)

    return M


def compute_rg_metrics(models, datasets, labels, cg_level_start, cg_level_end, logdir):

    metrics = {}

    metrics["keras"] = {}
    metrics["keras"]["evaluate_names"] = models.ediff.metrics_names
    for key in datasets:
        metrics["keras"]["ediff_loss_" + key] = models.ediff.evaluate(
            datasets[key][1], labels[key][1], verbose=0
        )

    metrics["rg"] = {}
    metrics["rg"]["cg_level_start"] = cg_level_start
    metrics["rg"]["cg_level_end"] = cg_level_end
    nn_basis = tf.keras.Model(
        inputs=models.energy.layers[0].input,
        outputs=models.energy.get_layer(name="sum_over_spins").output,
    )
    nn = {}
    nn["fine"] = nn_basis.predict(datasets["test"][cg_level_start])
    nn["coarse"] = nn_basis.predict(datasets["test"][cg_level_end])

    for key in nn:
        metrics["rg"]["sing_values_" + key] = np.linalg.svd(nn[key], compute_uv=False)
        metrics["rg"]["condition_num_" + key] = np.linalg.cond(nn[key])

    Mcc = compute_stat_avg(nn["coarse"], nn["coarse"])
    cc_cond = np.linalg.cond(Mcc)
    Mcf = compute_stat_avg(nn["coarse"], nn["fine"])

    J = np.linalg.lstsq(Mcc, Mcf, rcond=None)[0]
    J_eigs, _ = np.linalg.eig(J)
    criticalexp = np.log(2) / np.log(np.max(np.real(J_eigs)))
    metrics["condition_number_of_Mcc"] = cc_cond
    metrics["jacobian"] = J
    metrics["jacobian_eigs"] = J_eigs
    metrics["critical_exp"] = criticalexp

    metricfile = os.path.join(logdir, "metrics.json")
    with open(metricfile, "w") as outfile:
        json.dump(metrics, outfile)


def compute_mse(predictions, exact_labels):

    return np.mean(np.square(predictions - exact_labels))


def compute_exact_cg_metrics(model_group, datasets, labels, exact_labels, logdir):

    metrics = {}
    predict = {}

    for key in datasets:
        predict[key] = model_group.ediff.predict(datasets[key][1])
        metrics[key]["loss"] = compute_mse(predict[key], exact_labels[key])

        noise_train = labels[key][1] - exact_labels[key]
        metrics[key]["noise_var"] = np.var(noise_train)

    metricfile = os.path.join(logdir, "metrics_exact.json")
    with open(metricfile, "w") as outfile:
        json.dump(metrics, outfile)


def save_loss(history, logdir):

    min_loss = np.min(history.history["loss"])
    min_val_loss = np.min(history.history["val_loss"])
    shifted_loss = history.history["loss"]
    shifted_val_loss = history.history["val_loss"]

    plt.figure(1)
    plt.xlabel("epoch")
    plt.ylabel("loss, shifted")
    plt.yscale("log")
    plt.plot(shifted_loss, ".-", label="loss, min = {.5f}".format(min_loss))
    plt.plot(shifted_val_loss, ".-", label="val_loss, min = {.5f}".format(min_val_loss))
    plt.legend()
    plt.savefig(os.path.join(logdir, "loss.png"), bbox_inches="tight")

    lossfile = os.path.join(logdir, "loss.json")
    with open(lossfile, "w") as outfile:
        json.dump(history.history, outfile)
