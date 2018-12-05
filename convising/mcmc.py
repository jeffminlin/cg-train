import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import progressbar as pb


# wgt = [' [', pb.Timer(), '] ', pb.Bar(marker='█'), ' ', pb.Counter(), ' (', pb.Percentage(), ')', ' [', pb.AdaptiveETA(), '] ']


def metropolis_par(ediff_fun, tn, nsample, ss0, is_traj=False):
    """Run parallel metropolis chains

    Args:
        ediff_fun: Function which takes two lists of configurations and
            computes exp(-beta*(E2 - E1)) for each pair
        tn (int): Number of ``sweeps'', i.e. the number of proposed flips
        nsample (int): Interval at which to collect samples if is_traj=True
        ss0:  numpy array of shape (L^2,) or (nchains, L^2) to initalize with;
            L^2 is the number of spins in a configuration and nchains is the
            number of parallel metropolis chains to run
        is_traj (bool): Record the sample every nsample sweeps.
            Defaults to False.

    Returns:
        (final configuration, trajectory, acceptance ratio)

    """

    if np.array(ss0).ndim == 1:
        ss0 = np.array([ss0])
    N = ss0.shape[1]
    L = int(np.sqrt(N))
    nchains = ss0.shape[0]

    ss_traj = []
    if is_traj:
        ss_traj = np.zeros((int(np.ceil(tn/nsample)) * nchains, N))

    accept_count = 0
    sample_step = 0
    ss = np.copy(ss0)

    random_spin_flip = np.random.randint(N, size=(tn,nchains))

    # progress = pb.ProgressBar(max_value=tn, widgets=wgt).start()
    for tstep, ssidx in enumerate(random_spin_flip):
        prop_ss = np.copy(ss)
        prop_ss[range(nchains), ssidx] *= -1

        energy_diff = ediff_fun([ss.reshape((-1,L,L)), prop_ss.reshape((-1,L,L))]).reshape((nchains))
        flipidx = np.random.random(nchains) < energy_diff

        accept_count += np.count_nonzero(flipidx)
        ss[np.arange(nchains)[flipidx], ssidx[flipidx]] *= -1

        if tstep % nsample == 0:
            if is_traj:
                ss_traj[sample_step::int(np.ceil(tn/nsample)),:] = ss
            sample_step += 1

        # progress.update(tstep + 1)

    return (ss, ss_traj, accept_count / (tn * nchains))


def gen_samples(ediff_fun, tn, nsample, ss0, nburn):

    print()
    print("Burning phase")
    outobj_burn = metropolis_par(ediff_fun, nburn, nsample, ss0, is_traj=False)
    print()
    print("Acceptance rate:", outobj_burn[2])
    print()

    print("Running phase")
    outobj_run = metropolis_par(ediff_fun, tn, nsample, outobj_burn[0], is_traj=True)
    print()
    print("Acceptance rate:", outobj_run[2])
    print()

    return outobj_run[1]


def auto_corr_fast(M, kappa):

    # The autocorrelation has to be truncated at some point so there are enough data points constructing each lag. Let kappa be the cutoff
    M = M - np.mean(M)
    N = len(M)
    fvi = np.fft.fft(M, n=2*N)
    # G is the autocorrelation curve
    G = np.real(np.fft.ifft(fvi * np.conjugate(fvi))[:N])
    G /= N - np.arange(N)
    G /= G[0]
    G = G[:kappa]
    return G


def tau(M, kappa):

    # autocorr_fun is the unintegrated autocorrelation curve
    autocorr_fun = auto_corr_fast(M, kappa)
    # tau = 1 + 2*sum(G)
    return 1 + 2*np.sum(autocorr_fun), autocorr_fun


def plot_iac(Ms, labels, title, kappa, plotname):

    IACs = []
    Gs = []

    # loop through each chain
    for ind, M in enumerate(Ms):
        # get IAC and autocorrelation curve
        IAC, G = tau(M, kappa)
        IACs.append(IAC)
        Gs.append(G)
        plt.plot(G, label="{}: IAC = {:.2f}".\
                                            format(labels[ind], IAC))
    plt.legend(loc='best')
    plt.title(title)
    plt.tight_layout()
    #plt.show()
    plt.savefig(plotname)

    return IACs, Gs


class Observables:

    def __init__(self, images):
        self.numspins = images.shape[1] * images.shape[2]
        self.compute_neighbor_energy(images)
        self.compute_mag(images)

    def compute_neighbor_energy(self, images):
        self.first_nearest = np.sum(0.5 * (images * np.roll(images, 1, axis=1)
            + images * np.roll(images, -1, axis=1)
            + images * np.roll(images, 1, axis=2)
            + images * np.roll(images, -1, axis=2)), axis=(1,2))
        self.first_nearest /= self.numspins
        self.avg_first_nearest = np.mean(self.first_nearest)
        self.var_first_nearest = np.var(self.first_nearest)
        self.second_nearest = np.sum(0.5 * (images * np.roll(images, (1,1), axis=(1,2))
            + images * np.roll(images, (1,-1), axis=(1,2))
            + images * np.roll(images, (-1,1), axis=(1,2))
            + images * np.roll(images, (-1,-1), axis=(1,2))), axis=(1,2))
        self.second_nearest /= self.numspins
        self.avg_second_nearest = np.mean(self.second_nearest)
        self.var_second_nearest = np.var(self.second_nearest)
        self.four_spins = np.sum(images * np.roll(images, 1, axis=1)
            * np.roll(images, 1, axis=2)
            * np.roll(images, (1,1), axis=(1,2)), axis=(1,2))
        self.four_spins /= self.numspins
        self.avg_four_spins = np.mean(self.four_spins)
        self.var_four_spins = np.var(self.four_spins)

    def compute_mag(self, images):
        self.mag = np.sum(images, axis=(1,2)) / self.numspins
        self.avg_mag = np.mean(self.mag)
        self.var_mag = np.var(self.mag)

    def print_observables(self):
        num_samples = len(self.first_nearest)
        print()
        print("Number of samples:", num_samples)
        print()
        print("Nearest neighbor interaction (per spin):", self.avg_first_nearest)
        print("Variance of nn interaction (per spin):", self.var_first_nearest)
        print("Standard error, nn:", np.sqrt(self.var_first_nearest/num_samples))
        print()
        print("Second nn interaction (per spin):", self.avg_second_nearest)
        print("Variance of second nn interaction (per spin):", self.var_second_nearest)
        print("Standard error, second nn:", np.sqrt(self.var_second_nearest/num_samples))
        print("Four spin interaction (per spin):", self.avg_four_spins)
        print("Variance of four spin interaction (per spin):", self.var_four_spins)
        print("Standard error, four spin:", np.sqrt(self.var_four_spins/num_samples))
        print()
        print("Average magnetization (per spin):", self.avg_mag)
        print("Variance of magnetization (per spin):", self.var_mag)
        print("Standard error, mag:", np.sqrt(self.var_mag/num_samples))
        print()
