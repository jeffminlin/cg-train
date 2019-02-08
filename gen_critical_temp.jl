# Swendsen-Wang sampling of 2D Ising model
# Revision: 10/30/2018

include("mcmc_ising.jl")

using .MCMCIsing

N = 16
println("Size of original spin system = ", N)

beta = 0.4406868

tN = 5e6       # number of time steps in MCMC

coefTerm = [beta,0,0]
ss0 = rand(Int64,N,N)
nSample = 10
nstep = 10

using HDF5
using Printf
using DelimitedFiles

h5open("./data/L$(N)b$(@sprintf("%.4e", beta)).h5", "cw") do file

    d_ss = d_create(file, "images", Int8, ((floor(Int, nstep*tN/nSample),N*N),(-1,N*N)), "chunk", (2e4,N*N))
    d_e = d_create(file, "energies", Float64, ((floor(Int, nstep*tN/nSample),),(-1,)), "chunk", (2e4,))

    for step = 1:10
        println("Swendsen-Wang method")
        println("Step ", step)
        println("Burning phase..")
        @time outobj = MCMCIsing.ising_2d_sw(coefTerm, 10000, ss0, nSample,
                                            is_trajectory=false)
        println("Running phase..")
        @time outobj = MCMCIsing.ising_2d_sw(coefTerm,tN, outobj[4], nSample,
                                            is_trajectory=true)

        ss_traj = outobj[5]
        println(size(ss_traj))
        energy_traj = outobj[6]

        d_ss[floor(Int, (step-1)*tN/nSample+1):floor(Int, step*tN/nSample),:] = convert(Array{Int8,2}, ss_traj)
        d_e[floor(Int, (step-1)*tN/nSample+1):floor(Int, step*tN/nSample)] = energy_traj
    end
end
