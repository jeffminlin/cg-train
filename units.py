import numpy as np
import learn as ln

import neuralnets as nn


def check_cg(config):

    config.L = 4
    config.cg_method = "deci"
    config.cg_factor = 2
    testgen = nn.IsingCG(config)

    testimages = np.random.choice([-1,1], size=(3,4,4))
    teste = np.random.uniform(0.0, 1.0, size=3)

    ([testcg,testcgflip],testediff) = testgen._IsingCG__coarse_grain(testimages,teste)

    print(testimages)
    print(testcg)
    print(testcgflip)
    print(testediff)


def check_cg_e(config):

    config.L = 4
    config.cg_method = "deci"
    config.cg_factor = 2

    deep_conv = nn.ConvIsing(config)

    print(deep_conv.imagearray[10:12])
    print(deep_conv.earray[10:12])
    print(deep_conv.train_traj[0][10:12])
    print(deep_conv.train_traj[1][10:12])
    print(deep_conv.exp_ediff[10:12])
    print(deep_conv.cg_e[10:12])


def main():

    config = ln.Config()
    check_cg_e(config)


if __name__ == '__main__':

    main()
