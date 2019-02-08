import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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

    def __init__(self, ediff_fun, L, num_samples, num_chains, num_burn, skip):
        self.ediff_fun = ediff_fun
        self.numspins = L*L
        self.L = L
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.num_burn = num_burn
        self.skip = skip
        self.num_sweeps = (int(self.num_samples * self.skip / self.num_chains)
            + self.num_burn)

        self.avgs, self.vars = self.compute_observables(np.zeros((1, self.numspins)))
        self.num_recorded = 0

    def compute_observables(self, images, batch_size):
        avgs = np.zeros(4)
        vars = np.zeros(4)
        avgs_new = np.zeros(4)
        vars_new = np.zeros(4)
        num_computed = 0

        num_batches = int(np.floor(self.num_samples / float(batch_size)))

        for batchidx in range(num_batches):
            image = images[batchidx * batch_size:(batchidx + 1) * batch_size,:].reshape((-1, self.L, self.L))
            first_nearest = np.sum(0.5 * (image * np.roll(image, 1, axis=1)
                + image * np.roll(image, -1, axis=1)
                + image * np.roll(image, 1, axis=2)
                + image * np.roll(image, -1, axis=2)), axis=(1,2))
            avgs_new[0] = first_nearest / self.numspins
            second_nearest = np.sum(0.5 * (image * np.roll(image, (1,1), axis=(1,2))
                + image * np.roll(image, (1,-1), axis=(1,2))
                + image * np.roll(image, (-1,1), axis=(1,2))
                + image * np.roll(image, (-1,-1), axis=(1,2))), axis=(1,2))
            avgs_new[1] = second_nearest / self.numspins
            four_spins = np.sum(image * np.roll(image, 1, axis=1)
                * np.roll(image, 1, axis=2)
                * np.roll(image, (1,1), axis=(1,2)), axis=(1,2))
            avgs_new[2] = four_spins / self.numspins
            mag = np.sum(image, axis=(1,2))
            avgs_new[3] = mag / self.numspins
            avgs, vars, num_computed = self.update_observables(avgs, vars, num_computed, avgs_new, vars_new, image.shape[0])

        return avgs, vars

    def metrop_par(self, batch_size):
        accept_count = 0
        ss = np.random.choice([-1, 1], size=(self.num_chains,
                                             self.numspins))

        random_spin_flip = np.random.randint(self.numspins, size=(self.num_sweeps,
            self.num_chains))

        for tstep, ssidx in enumerate(random_spin_flip):
            prop_ss = np.copy(ss)
            prop_ss[range(self.num_chains), ssidx] *= -1

            energy_diff = self.ediff_fun([ss.reshape((-1, self.L, self.L)), prop_ss.reshape((-1, self.L, self.L))]).reshape((self.num_chains))
            flipidx = np.random.random(self.num_chains) < energy_diff

            accept_count += np.count_nonzero(flipidx)
            ss[np.arange(self.num_chains)[flipidx], ssidx[flipidx]] *= -1

            if (tstep % self.skip == 0) and (tstep > self.num_burn):
                # Compute observables
                avgs, vars = self.compute_observables(ss, batch_size)
                self.avgs, self.vars, self.num_recorded = self.update_observables(self.avgs, self.vars, self.num_recorded, avgs, vars, self.num_chains)

        return accept_count / (self.num_sweeps * self.num_chains)

    def update_observables(self, avgs_old, vars_old, num_old, avgs_new, vars_new, num_new):
        # Parallel updates
        num_tot = num_old + num_new
        avgs = (avgs_old * num_old + avgs_new * num_new) / num_tot

        delta = avgs_new - avgs_old
        M2_old = vars_old * (num_old - 1.0)
        M2_new = vars_new * (num_new - 1.0)
        M2 = M2_old + M2_new + np.square(delta) * num_old * num_new / num_tot
        if num_tot > 1:
            vars = M2 / (num_tot - 1.0)
        else:
            vars = M2

        return avgs, vars, num_tot

    def print_observables(self):
        print()
        print("Number of samples:", self.num_recorded)
        print()
        print("Nearest neighbor interaction (per spin):", self.avgs[0])
        print("Variance of nn interaction (per spin):", self.vars[0])
        print("Standard error (biased), nn:", np.sqrt(self.vars[0] / self.num_recorded))
        print()
        print("Second nn interaction (per spin):", self.avgs[1])
        print("Variance of second nn interaction (per spin):", self.vars[1])
        print("Standard error (biased), second nn:", np.sqrt(self.vars[1] / self.num_recorded))
        print()
        print("Four spin interaction (per spin):", self.avgs[2])
        print("Variance of four spin interaction (per spin):", self.vars[2])
        print("Standard error (biased), four spin:", np.sqrt(self.vars[2] / self.num_recorded))
        print()
        print("Average magnetization (per spin):", self.avgs[3])
        print("Variance of magnetization (per spin):", self.vars[3])
        print("Standard error (biased), mag:", np.sqrt(self.vars[3] / self.num_recorded))
        print()
