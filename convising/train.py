"""Module for training"""
import os
import json

import h5py
import matplotlib

matplotlib.use("Agg")  # matplotlib backend for HPC
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import convising.models as models


def create_default_config_ising():

    return {"L": 8, "beta": 0.4406868, "cg_method": "maj", "cg_factor": 2}


def create_default_config_train():

    config = {
        "keep_data": 1.0,
        "train_split": 9.0 / 10.0,
        "val_split": 1.0 / 9.0,
        "batch_size": 20000,
        "nepochs": 1000,
        "patience": 100,
        "verbosity": 0,
        "conv_activation": models.log_cosh,
        "dense_activation": "elu",
        "kernel_size": 3,
        "nfilters": 4,
        "dense_nodes": [20, 20, 4],
    }

    return config


def train(model_group, datasets, labels, config_train, cg_level, logdir, freeze=False):

    os.makedirs(logdir, exist_ok=True)
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
        monitor="val_loss", patience=config_train["patience"], restore_best_weights=True
    )
    callback_list = [best_weight, early_stop]

    # Train neural net
    if datasets[cg_level].get("val"):
        val_data = (datasets[cg_level]["val"], labels["val"])
    else:
        val_data = None

    history = model_group.ediff.fit(
        x=datasets[cg_level]["train"],
        y=labels["train"],
        verbose=config_train["verbosity"],
        batch_size=config_train["batch_size"],
        validation_data=val_data,
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

    print("Training")
    history = train(
        model_group, datasets, labels, config_train, cg_level_start, logdir, freeze
    )
    print("Saving training history")
    save_loss(history, logdir)
    print("Saving metrics")
    compute_rg_metrics(
        model_group, datasets, labels, cg_level_start, cg_level_end, logdir
    )
    if exact_labels:
        print("Saving exact metrics")
        compute_exact_cg_metrics(model_group, datasets, labels, exact_labels, logdir)

    modelpath = os.path.join(logdir, "model")
    model_group.energy.save(modelpath)


def compute_stat_avg(first_samples, second_samples):

    average_first = np.average(first_samples, axis=0).reshape(1, -1)
    average_second = np.average(second_samples, axis=0).reshape(1, -1)
    M = np.matmul(first_samples.transpose(), second_samples) / second_samples.shape[0]
    M = M - np.matmul(average_first.transpose(), average_second)

    return M


def compute_rg_metrics(models, datasets, labels, cg_level_start, cg_level_end, logdir):

    metrics = {}

    if datasets.get(1):
        metrics["keras"] = {}
        metrics["keras"]["evaluate_names"] = models.ediff.metrics_names
        for key in datasets[1]:
            metrics["keras"]["ediff_loss_" + key] = models.ediff.evaluate(
                datasets[1][key], labels[key], verbose=0
            )

    metrics["rg"] = {}
    metrics["rg"]["cg_level_start"] = cg_level_start
    metrics["rg"]["cg_level_end"] = cg_level_end
    nn_basis = tf.keras.Model(
        inputs=models.energy.layers[0].input,
        outputs=models.energy.get_layer(name="sum_over_spins").output,
    )
    nn = {}
    nn["fine"] = nn_basis.predict(datasets[cg_level_start]["test"])
    nn["coarse"] = nn_basis.predict(datasets[cg_level_end]["test"])

    for key in nn:
        metrics["rg"]["sing_values_" + key] = np.linalg.svd(
            nn[key], compute_uv=False
        ).tolist()
        metrics["rg"]["condition_num_" + key] = np.linalg.cond(nn[key]).tolist()

    Mcc = compute_stat_avg(nn["coarse"], nn["coarse"])
    cc_cond = np.linalg.cond(Mcc)
    Mcf = compute_stat_avg(nn["coarse"], nn["fine"])

    J = np.linalg.lstsq(Mcc, Mcf, rcond=None)[0]
    J_eigs, _ = np.linalg.eig(J)
    criticalexp = np.log(2) / np.log(np.max(np.real(J_eigs)))
    metrics["condition_number_of_Mcc"] = float(cc_cond)
    metrics["jacobian"] = J.tolist()
    metrics["jacobian_eigs"] = J_eigs.tolist()
    metrics["critical_exp"] = float(criticalexp)

    with open_or_create(logdir, "metrics.json", "w") as outfile:
        json.dump(metrics, outfile)


def compute_mse(predictions, exact_labels):

    return np.mean(np.square(predictions - exact_labels))


def compute_exact_cg_metrics(model_group, datasets, labels, exact_labels, logdir):

    metrics = {}
    predict = {}

    for key in datasets:
        predict[key] = model_group.ediff.predict(datasets[key][1])
        metrics[key]["loss"] = float(compute_mse(predict[key], exact_labels[key]))

        noise = labels[key] - exact_labels[key]
        metrics[key]["noise_var"] = float(np.var(noise))

    with open_or_create(logdir, "metrics_exact.json", "w") as outfile:
        json.dump(metrics, outfile)


def save_loss(history, logdir):

    min_loss = np.min(history.history["loss"])
    min_val_loss = np.min(history.history["val_loss"])
    shifted_loss = history.history["loss"] - min_loss
    shifted_val_loss = history.history["val_loss"] - min_val_loss

    plt.figure(1)
    plt.xlabel("epoch")
    plt.ylabel("loss, shifted")
    plt.yscale("log")
    plt.plot(shifted_loss, ".-", label="loss, min = {:.5f}".format(min_loss))
    plt.plot(
        shifted_val_loss, ".-", label="val_loss, min = {:.5f}".format(min_val_loss)
    )
    plt.legend()
    plt.savefig(os.path.join(logdir, "loss.png"), bbox_inches="tight")

    with open_or_create(logdir, "loss.json", "w") as outfile:
        json.dump(history.history, outfile)


def open_or_create(path, filename, option):

    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    return open(filepath, option)
