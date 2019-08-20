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


def compute_rg_metrics(
    model_group, datasets, labels, cg_level_start, cg_level_end, logdir
):

    metrics = {}

    metrics["keras"] = {}
    metrics["keras"]["evaluate_names"] = model_group.ediff.metrics_names
    metrics["keras"]["ediff_loss_train"] = model_group.ediff.evaluate(
        datasets["train"][1], labels["train"][1], verbose=0
    )
    metrics["keras"]["ediff_loss_test"] = model_group.ediff.evaluate(
        datasets["test"][1], labels["test"][1], verbose=0
    )

    metrics["rg"] = {}
    metrics["rg"]["cg_level_start"] = cg_level_start
    metrics["rg"]["cg_level_end"] = cg_level_end
    nn_basis = tf.keras.Model(
        inputs=model_group.energy.layers[0].input,
        outputs=model_group.energy.get_layer(name="sum_over_spins").output,
    )
    nn_fine = nn_basis.predict(datasets["test"][cg_level_start])
    nn_coarse = nn_basis.predict(datasets["test"][cg_level_end])

    metrics["rg"]["sing_values_fine"] = np.linalg.svd(nn_fine, compute_uv=False)
    metrics["rg"]["condition_num_fine"] = np.linalg.cond(nn_fine)

    metrics["rg"]["sing_values_coarse"] = np.linalg.svd(nn_coarse, compute_uv=False)
    metrics["rg"]["condition_num_coarse"] = np.linalg.cond(nn_coarse)

    coarse_avg = np.average(nn_coarse, axis=0)
    coarse_avg = coarse_avg.reshape(1, -1)

    fine_avg = np.average(nn_fine, axis=0)
    fine_avg = fine_avg.reshape(1, -1)

    Mcc = np.matmul(nn_coarse.transpose(), nn_coarse) / nn_coarse.shape[0] - np.matmul(
        coarse_avg.transpose(), coarse_avg
    )
    cc_cond = np.linalg.cond(Mcc)
    Mcf = np.matmul(nn_coarse.transpose(), nn_fine) / nn_coarse.shape[0] - np.matmul(
        coarse_avg.transpose(), fine_avg
    )

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


def compute_exact_cg_metrics(model_group, datasets, labels, exact_labels, logdir):

    metrics = {}

    predict_train = model_group.ediff.predict(datasets["train"][1])
    predict_test = model_group.ediff.predict(datasets["test"][1])
    metrics["train"]["loss"] = np.mean(np.square(predict_train - exact_labels["train"]))
    metrics["test"]["loss"] = np.mean(np.square(predict_test - exact_labels["test"]))

    noise_train = labels["train"][1] - exact_labels["train"]
    noise_test = labels["test"][1] - exact_labels["test"]
    metrics["train"]["noise_var"] = np.var(noise_train)
    metrics["test"]["noise_var"] = np.var(noise_test)

    metricfile = os.path.join(logdir, "metrics_exact.json")
    with open(metricfile, "w") as outfile:
        json.dump(metrics, outfile)


def graph_loss(history, logdir):
    plt.figure(1)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.plot(
        history.history["loss"] - np.min(history.history["loss"]), ".-", label="loss"
    )
    plt.plot(
        history.history["val_loss"] - np.min(history.history["val_loss"]),
        ".-",
        label="val_loss",
    )
    plt.legend()
    plt.savefig(os.path.join(logdir, "loss.png"), bbox_inches="tight")
