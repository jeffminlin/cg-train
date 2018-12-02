import numpy as np

import convising.train as tr
import convising.mcmc as mc


def run():

    config = tr.Config()
    config.L = 8
    deep_conv = tr.ConvIsing(config)

    deep_conv.create_dataset(config)
    deep_conv.run_model(config)
    deep_conv.compute_metrics()
    deep_conv.print_metrics()
    deep_conv.graph_loss(config)


def compute_iac(deep_conv, config):

    ss_traj = mc.gen_samples(deep_conv.model, int(1e4), 1, np.random.choice([-1,1], size=(100,config.cgL*config.cgL)), 5000)
    e_traj = deep_conv.model_energy.predict(ss_traj.reshape([-1,config.cgL,config.cgL])).ravel()

    kappa = 250
    plotname = "".join(["./figs/autocorr", config.filepath, ".png"])
    IAC, autocorr_fun = mc.plot_iac([e_traj], ["Standard M-H"], "".join(["Metropolis-Hastings, L = ", str(config.L), ", CG L = ", str(config.cgL)]), kappa, plotname)
    print("Integrated autocorrelation, kappa =", kappa, ":", IAC)


def compare_observables(deep_conv, config):

    ss_traj = mc.gen_samples(deep_conv.model, int(1e5), 100, np.random.choice([-1,1], size=(1000,config.cgL*config.cgL)), 5000)
    obs_model = mc.Observables(ss_traj.reshape([-1,config.cgL,config.cgL]))
    obs_samples = mc.Observables(np.loadtxt(config.imagefile).reshape([-1,config.L,config.L])[:,0::config.cg_factor,0::config.cg_factor])

    print()
    print("Samples generated using learned model")
    obs_model.print_observables()
    print("Samples generated with Swendsen-Wang")
    obs_samples.print_observables()


def project_model(deep_conv, config):

    samples = np.loadtxt(config.imagefile).reshape([-1,config.L,config.L])[:,0::config.cg_factor,0::config.cg_factor]
    obs = mc.Observables(samples)
    e_traj = deep_conv.model_energy.predict(samples)
    projection = np.linalg.lstsq(obs.numspins * np.column_stack((np.ones(len(samples)), obs.first_nearest, obs.second_nearest, obs.four_spins)), e_traj, rcond=None)

    print()
    print("Projection onto constant, first nn, second nn, and four spin products:")
    print(projection[0])
    print()
    print("Norm of constant:", np.linalg.norm(obs.numspins * np.ones(len(samples))))
    print("Norm of first nn:", np.linalg.norm(obs.numspins * obs.first_nearest))
    print("Norm of second nn:", np.linalg.norm(obs.numspins * obs.first_nearest))
    print("Norm of four spin products:", np.linalg.norm(obs.numspins * obs.four_spins))
    print("Norm of model energy:", np.linalg.norm(e_traj))
    print("Norm of residual:", np.sqrt(projection[1][0]))
    print("R^2:", 1 - projection[1][0] / (len(e_traj) * np.var(e_traj)))
    print()


def main():

    config = tr.Config()
    deep_conv = tr.ConvIsing(config)
    deep_conv.reload_weights(config)
    compute_iac(deep_conv, config)


if __name__ == '__main__':

    main()
