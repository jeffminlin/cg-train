# Swendsen-Wang sampling of 2D Ising model
# Revision: 10/30/2018

include("mcmc_ising.jl")

using .MCMCIsing

N = 4
println("Size of original spin system = ", N)

beta = 0.4406868

tN = 5e6       # number of time steps in MCMC

coefTerm = [beta,0,0]
ss0 = rand(Int64,N,N)
nSample = 10


println("Swendsen-Wang method")
println("Burning phase..")
@time outobj = MCMCIsing.ising_2d_sw(coefTerm,10000, ss0, nSample,
                                    is_trajectory=false )
println("Running phase..")
@time outobj = MCMCIsing.ising_2d_sw(coefTerm,tN, outobj[4], nSample,
                                    is_trajectory = true)

ss_traj = outobj[5]
energy_traj = outobj[6]

using Printf
using DelimitedFiles

f_ss = open("./data/ssL$(N)b$(@sprintf("%.4e", beta)).dat", "a")
f_e = open("./data/EL$(N)b$(@sprintf("%.4e", beta)).dat", "a")
writedlm(f_ss, transpose(ss_traj))
writedlm(f_e, energy_traj)
close(f_ss)
close(f_e)
